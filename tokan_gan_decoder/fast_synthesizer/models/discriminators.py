"""
discriminators.py: Multi-scale and multi-period discriminators for GAN-based mel decoder.

Based on HiFi-GAN discriminator architecture, adapted for mel spectrogram domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DiscriminatorP(nn.Module):
    """
    Period Discriminator.
    
    Operates on periodic sub-sampled patterns of the mel spectrogram.
    """
    
    def __init__(
        self,
        period: int,
        n_mel: int = 80,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.period = period
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(n_mel, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Mel spectrogram (B, n_mel, T)
            
        Returns:
            output: Discriminator output
            features: Intermediate feature maps for feature matching loss
        """
        features = []
        
        # Reshape to 2D format with period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = x.flatten(1, -1)
        
        return x, features


class DiscriminatorS(nn.Module):
    """
    Scale Discriminator.
    
    Operates at different scales of the mel spectrogram.
    """
    
    def __init__(
        self,
        n_mel: int = 80,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(n_mel, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Mel spectrogram (B, n_mel, T)
            
        Returns:
            output: Discriminator output
            features: Intermediate feature maps
        """
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        x = x.flatten(1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator combining multiple period discriminators.
    """
    
    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        n_mel: int = 80
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period, n_mel=n_mel) for period in periods
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Args:
            y: Real mel spectrogram (B, n_mel, T)
            y_hat: Generated mel spectrogram (B, n_mel, T)
            
        Returns:
            y_d_rs: Discriminator outputs for real
            y_d_gs: Discriminator outputs for generated
            fmap_rs: Feature maps for real
            fmap_gs: Feature maps for generated
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator combining multiple scale discriminators.
    """
    
    def __init__(
        self,
        n_scales: int = 3,
        n_mel: int = 80
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            DiscriminatorS(n_mel=n_mel, use_spectral_norm=(i == 0))
            for i in range(n_scales)
        ])
        
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2)
            for _ in range(n_scales - 1)
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Args:
            y: Real mel spectrogram (B, n_mel, T)
            y_hat: Generated mel spectrogram (B, n_mel, T)
            
        Returns:
            y_d_rs: Discriminator outputs for real
            y_d_gs: Discriminator outputs for generated
            fmap_rs: Feature maps for real
            fmap_gs: Feature maps for generated
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                y = self.pools[i - 1](y)
                y_hat = self.pools[i - 1](y_hat)
            
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """
    Combined discriminator using both multi-period and multi-scale discriminators.
    """
    
    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        n_scales: int = 3,
        n_mel: int = 80
    ):
        super().__init__()
        
        self.mpd = MultiPeriodDiscriminator(periods=periods, n_mel=n_mel)
        self.msd = MultiScaleDiscriminator(n_scales=n_scales, n_mel=n_mel)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> dict:
        """
        Args:
            y: Real mel spectrogram (B, n_mel, T)
            y_hat: Generated mel spectrogram (B, n_mel, T)
            
        Returns:
            Dictionary containing all discriminator outputs and feature maps
        """
        # Multi-period discriminator
        mpd_y_d_rs, mpd_y_d_gs, mpd_fmap_rs, mpd_fmap_gs = self.mpd(y, y_hat)
        
        # Multi-scale discriminator
        msd_y_d_rs, msd_y_d_gs, msd_fmap_rs, msd_fmap_gs = self.msd(y, y_hat)
        
        return {
            "mpd": {
                "real": mpd_y_d_rs,
                "fake": mpd_y_d_gs,
                "fmap_real": mpd_fmap_rs,
                "fmap_fake": mpd_fmap_gs
            },
            "msd": {
                "real": msd_y_d_rs,
                "fake": msd_y_d_gs,
                "fmap_real": msd_fmap_rs,
                "fmap_fake": msd_fmap_gs
            }
        }


# =============================================================================
# Lightweight Discriminator (for faster training)
# =============================================================================

class LightweightDiscriminator(nn.Module):
    """
    Lightweight discriminator for faster training.
    
    Single multi-scale discriminator with fewer parameters.
    """
    
    def __init__(self, n_mel: int = 80):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv1d(n_mel, 64, 15, 1, padding=7)),
            nn.utils.spectral_norm(nn.Conv1d(64, 128, 41, 4, groups=4, padding=20)),
            nn.utils.spectral_norm(nn.Conv1d(128, 256, 41, 4, groups=8, padding=20)),
            nn.utils.spectral_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.utils.spectral_norm(nn.Conv1d(512, 512, 5, 1, padding=2)),
        ])
        
        self.conv_post = nn.utils.spectral_norm(nn.Conv1d(512, 1, 3, 1, padding=1))

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            y: Real mel spectrogram (B, n_mel, T)
            y_hat: Generated mel spectrogram (B, n_mel, T)
            
        Returns:
            d_real: Discriminator output for real
            d_fake: Discriminator output for generated
            fmap_real: Feature maps for real
            fmap_fake: Feature maps for generated
        """
        fmap_real = []
        fmap_fake = []
        
        x_real = y
        x_fake = y_hat
        
        for conv in self.convs:
            x_real = F.leaky_relu(conv(x_real), 0.1)
            x_fake = F.leaky_relu(conv(x_fake), 0.1)
            fmap_real.append(x_real)
            fmap_fake.append(x_fake)
        
        d_real = self.conv_post(x_real)
        d_fake = self.conv_post(x_fake)
        
        return d_real, d_fake, fmap_real, fmap_fake


if __name__ == "__main__":
    # Test discriminators
    batch_size = 2
    n_mel = 80
    seq_len = 100
    
    y = torch.randn(batch_size, n_mel, seq_len)
    y_hat = torch.randn(batch_size, n_mel, seq_len)
    
    # Test combined discriminator
    print("Testing CombinedDiscriminator...")
    disc = CombinedDiscriminator(n_mel=n_mel)
    outputs = disc(y, y_hat)
    print(f"  MPD outputs: {len(outputs['mpd']['real'])}")
    print(f"  MSD outputs: {len(outputs['msd']['real'])}")
    print(f"  Parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # Test lightweight discriminator
    print("\nTesting LightweightDiscriminator...")
    disc = LightweightDiscriminator(n_mel=n_mel)
    d_real, d_fake, fmap_real, fmap_fake = disc(y, y_hat)
    print(f"  Output shape: {d_real.shape}")
    print(f"  Feature maps: {len(fmap_real)}")
    print(f"  Parameters: {sum(p.numel() for p in disc.parameters()):,}")
