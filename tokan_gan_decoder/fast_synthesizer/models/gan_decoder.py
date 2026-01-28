"""
gan_decoder.py: Single-Pass GAN-based Decoder for TokAN

This decoder replaces the iterative CFM decoder with a single forward pass
convolutional generator for real-time mel spectrogram synthesis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange


# =============================================================================
# Building Blocks
# =============================================================================

class ConvNorm(nn.Module):
    """1D Convolution with weight normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = 'linear'
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size * dilation - dilation) // 2
        
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions (HiFi-GAN style)."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
        lrelu_slope: float = 0.1
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for dilation in dilations:
            self.convs1.append(
                ConvNorm(channels, channels, kernel_size, dilation=dilation)
            )
            self.convs2.append(
                ConvNorm(channels, channels, kernel_size, dilation=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x


class ConvTransposeBlock(nn.Module):
    """Transposed convolution block for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        lrelu_slope: float = 0.1
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        padding = (kernel_size - stride) // 2
        
        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, self.lrelu_slope)
        return self.conv(x)


# =============================================================================
# Main Generator Architecture
# =============================================================================

class GANMelDecoder(nn.Module):
    """
    Single-pass GAN-based mel spectrogram decoder.
    
    Takes aligned encoder features h_y and generates mel spectrograms
    in a single forward pass.
    
    Architecture inspired by HiFi-GAN generator and FastSpeech2 decoder.
    
    Args:
        in_channels: Input feature dimension from encoder (default: 256)
        out_channels: Output mel spectrogram channels (default: 80)
        hidden_channels: Hidden dimension (default: 512)
        kernel_sizes: Kernel sizes for each residual stack (default: (3, 7, 11))
        dilations: Dilations for each residual block (default: ((1, 3, 5), (1, 3, 5), (1, 3, 5)))
        n_res_blocks: Number of residual blocks per stack (default: 3)
        spk_emb_dim: Speaker embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 80,
        hidden_channels: int = 512,
        kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        n_res_blocks: int = 3,
        spk_emb_dim: int = 256,
        dropout: float = 0.1,
        lrelu_slope: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lrelu_slope = lrelu_slope
        
        # Input projection
        self.input_proj = ConvNorm(in_channels, hidden_channels, kernel_size=7)
        
        # Optional speaker conditioning (additional to what's in h_y)
        self.spk_proj = nn.Linear(spk_emb_dim, hidden_channels) if spk_emb_dim > 0 else None
        
        # Main decoder stack
        self.res_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            for kernel_size, dilation in zip(kernel_sizes, dilations):
                self.res_blocks.append(
                    ResBlock(hidden_channels, kernel_size, dilation, lrelu_slope)
                )
        
        # PostNet for mel refinement
        self.postnet = PostNet(out_channels, hidden_channels=256, n_layers=5, kernel_size=5)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(lrelu_slope),
            ConvNorm(hidden_channels, out_channels, kernel_size=7),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        h_y: torch.Tensor,
        y_mask: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for mel generation.
        
        Args:
            h_y: Aligned encoder features (B, D, T)
            y_mask: Mel mask (B, 1, T)
            spks: Optional speaker embedding (B, spk_emb_dim)
            
        Returns:
            mel: Generated mel spectrogram (B, 80, T)
        """
        # Input projection
        x = self.input_proj(h_y) * y_mask
        
        # Optional additional speaker conditioning
        if self.spk_proj is not None and spks is not None:
            spk_emb = self.spk_proj(spks).unsqueeze(-1)
            x = x + spk_emb * y_mask
        
        x = self.dropout(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x) * y_mask
        
        # Output projection (coarse mel)
        mel_coarse = self.output_proj(x) * y_mask
        
        # PostNet refinement
        mel_refined = mel_coarse + self.postnet(mel_coarse) * y_mask
        
        return mel_refined, mel_coarse


class PostNet(nn.Module):
    """
    PostNet for mel spectrogram refinement.
    
    5-layer 1D convolution network that predicts a residual
    to add to the decoder output.
    """
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        hidden_channels: int = 256,
        n_layers: int = 5,
        kernel_size: int = 5
    ):
        super().__init__()
        
        self.convolutions = nn.ModuleList()
        
        # First layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, hidden_channels, kernel_size),
                nn.BatchNorm1d(hidden_channels)
            )
        )
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hidden_channels, hidden_channels, kernel_size),
                    nn.BatchNorm1d(hidden_channels)
                )
            )
        
        # Final layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hidden_channels, n_mel_channels, kernel_size),
                nn.BatchNorm1d(n_mel_channels)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = torch.tanh(conv(x))
            else:
                x = conv(x)
        return x


# =============================================================================
# Alternative: Conformer-based Decoder
# =============================================================================

class ConformerMelDecoder(nn.Module):
    """
    Conformer-based mel decoder for higher quality at slightly slower speed.
    
    Uses self-attention + convolution for better long-range modeling.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 80,
        hidden_channels: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        spk_emb_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Speaker projection
        self.spk_proj = nn.Linear(spk_emb_dim, hidden_channels) if spk_emb_dim > 0 else None
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(hidden_channels)
        
        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_channels, n_heads, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # PostNet
        self.postnet = PostNet(out_channels)

    def forward(
        self,
        h_y: torch.Tensor,
        y_mask: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            h_y: (B, D, T)
            y_mask: (B, 1, T)
            spks: (B, spk_emb_dim)
        
        Returns:
            mel_refined: (B, 80, T)
            mel_coarse: (B, 80, T)
        """
        # (B, D, T) -> (B, T, D)
        x = h_y.transpose(1, 2)
        mask = y_mask.squeeze(1)  # (B, T)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add speaker embedding
        if self.spk_proj is not None and spks is not None:
            spk_emb = self.spk_proj(spks).unsqueeze(1)  # (B, 1, D)
            x = x + spk_emb
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Conformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output projection
        mel_coarse = self.output_proj(x).transpose(1, 2)  # (B, 80, T)
        
        # PostNet refinement
        mel_refined = mel_coarse + self.postnet(mel_coarse) * y_mask
        
        return mel_refined, mel_coarse


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ffn1 = FeedForward(d_model, d_model * 4, dropout)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, d_model * 4, dropout)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class FeedForward(nn.Module):
    """Feed-forward module."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.silu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        
        x = self.norm(x)
        
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(~mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.pointwise1(x)
        x = F.glu(x, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        x = self.pointwise2(x)
        return self.dropout(x)


# =============================================================================
# Factory Function
# =============================================================================

def get_gan_decoder(
    decoder_type: str = "conv",
    **kwargs
) -> nn.Module:
    """
    Factory function to create GAN decoder.
    
    Args:
        decoder_type: "conv" for convolutional, "conformer" for conformer-based
        **kwargs: Additional arguments for the decoder
        
    Returns:
        Decoder module
    """
    if decoder_type == "conv":
        return GANMelDecoder(**kwargs)
    elif decoder_type == "conformer":
        return ConformerMelDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


if __name__ == "__main__":
    # Test the decoder
    batch_size = 2
    seq_len = 100
    in_channels = 256
    
    # Create inputs
    h_y = torch.randn(batch_size, in_channels, seq_len)
    y_mask = torch.ones(batch_size, 1, seq_len)
    spks = torch.randn(batch_size, 256)
    
    # Test convolutional decoder
    print("Testing GANMelDecoder...")
    decoder = GANMelDecoder(in_channels=in_channels)
    mel_refined, mel_coarse = decoder(h_y, y_mask, spks)
    print(f"  Input shape: {h_y.shape}")
    print(f"  Output shape: {mel_refined.shape}")
    print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Test conformer decoder
    print("\nTesting ConformerMelDecoder...")
    decoder = ConformerMelDecoder(in_channels=in_channels)
    mel_refined, mel_coarse = decoder(h_y, y_mask, spks)
    print(f"  Input shape: {h_y.shape}")
    print(f"  Output shape: {mel_refined.shape}")
    print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")
