"""
losses.py: Loss functions for GAN-based mel decoder training.

Includes:
- Multi-Resolution STFT Loss
- Adversarial Loss (LSGAN)
- Feature Matching Loss
- Mel Reconstruction Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# =============================================================================
# Multi-Resolution STFT Loss
# =============================================================================

class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss.
    
    Combines spectral convergence loss and log magnitude loss.
    """
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        
        # Register window as buffer
        self.register_buffer('window', torch.hann_window(win_size))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude."""
        # x: (B, T)
        x_stft = torch.stft(
            x,
            self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True
        )
        # Return magnitude
        return torch.abs(x_stft)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT loss between two mel spectrograms.
        
        Args:
            x: Generated mel (B, n_mel, T)
            y: Target mel (B, n_mel, T)
            
        Returns:
            sc_loss: Spectral convergence loss
            mag_loss: Log magnitude loss
        """
        # Flatten mel to simulate waveform-like signal for STFT
        # Actually, for mel domain, we use different approach
        
        # For mel spectrograms, compute losses directly on mel
        # Spectral convergence on mel
        sc_loss = torch.norm(y - x, p='fro') / (torch.norm(y, p='fro') + 1e-8)
        
        # Log magnitude loss on mel
        log_x = torch.log(torch.clamp(x, min=1e-7))
        log_y = torch.log(torch.clamp(y, min=1e-7))
        mag_loss = F.l1_loss(log_x, log_y)
        
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for mel spectrograms.
    
    Operates at multiple time-frequency resolutions.
    """
    
    def __init__(
        self,
        resolutions: List[Tuple[int, int, int]] = [
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024),
        ]
    ):
        super().__init__()
        
        self.losses = nn.ModuleList([
            STFTLoss(fft, hop, win) for fft, hop, win in resolutions
        ])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            x: Generated mel (B, n_mel, T)
            y: Target mel (B, n_mel, T)
            
        Returns:
            Total multi-resolution loss
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        for loss_fn in self.losses:
            sc_l, mag_l = loss_fn(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        
        return (sc_loss + mag_loss) / len(self.losses)


# =============================================================================
# Mel-domain Multi-resolution Loss
# =============================================================================

class MelMultiResolutionLoss(nn.Module):
    """
    Multi-resolution loss in mel domain.
    
    Applies pooling at different scales and computes L1 loss.
    """
    
    def __init__(self, scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        
        self.pools = nn.ModuleList([
            nn.AvgPool1d(scale, scale) if scale > 1 else nn.Identity()
            for scale in scales
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution mel loss.
        
        Args:
            x: Generated mel (B, n_mel, T)
            y: Target mel (B, n_mel, T)
            
        Returns:
            Total multi-resolution loss
        """
        total_loss = 0.0
        
        for pool in self.pools:
            x_pooled = pool(x)
            y_pooled = pool(y)
            total_loss += F.l1_loss(x_pooled, y_pooled)
        
        return total_loss / len(self.scales)


# =============================================================================
# Adversarial Losses
# =============================================================================

def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    LSGAN discriminator loss.
    
    Args:
        disc_real_outputs: List of discriminator outputs for real samples
        disc_generated_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Total discriminator loss
    """
    loss = 0.0
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
    
    return loss


def generator_loss(
    disc_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    LSGAN generator loss.
    
    Args:
        disc_outputs: List of discriminator outputs for generated samples
        
    Returns:
        Total generator adversarial loss
    """
    loss = 0.0
    
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    
    return loss


def feature_matching_loss(
    fmap_real: List[List[torch.Tensor]],
    fmap_fake: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    Feature matching loss.
    
    Matches intermediate features between real and generated samples.
    
    Args:
        fmap_real: List of feature map lists for real samples
        fmap_fake: List of feature map lists for generated samples
        
    Returns:
        Total feature matching loss
    """
    loss = 0.0
    
    for dr, dg in zip(fmap_real, fmap_fake):
        for r, g in zip(dr, dg):
            loss += F.l1_loss(r, g)
    
    return loss * 2  # Scale factor from HiFi-GAN


# =============================================================================
# Combined Loss Functions
# =============================================================================

class GANLoss(nn.Module):
    """
    Combined GAN loss for decoder training.
    """
    
    def __init__(
        self,
        lambda_mel: float = 45.0,
        lambda_fm: float = 2.0,
        lambda_mr: float = 1.0,
    ):
        super().__init__()
        
        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm
        self.lambda_mr = lambda_mr
        
        self.multi_res_loss = MelMultiResolutionLoss()

    def compute_generator_loss(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        disc_outputs: dict,
        mel_coarse: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute generator losses.
        
        Args:
            mel_pred: Predicted mel spectrogram (B, n_mel, T)
            mel_target: Target mel spectrogram (B, n_mel, T)
            disc_outputs: Dictionary of discriminator outputs
            mel_coarse: Optional coarse mel prediction
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Mel reconstruction loss
        mel_loss = F.l1_loss(mel_pred, mel_target)
        losses['mel_loss'] = mel_loss * self.lambda_mel
        
        # Multi-resolution loss
        mr_loss = self.multi_res_loss(mel_pred, mel_target)
        losses['mr_loss'] = mr_loss * self.lambda_mr
        
        # Coarse mel loss (if using postnet)
        if mel_coarse is not None:
            coarse_loss = F.l1_loss(mel_coarse, mel_target)
            losses['coarse_loss'] = coarse_loss * self.lambda_mel * 0.5
        
        # Adversarial losses
        g_loss_mpd = generator_loss(disc_outputs['mpd']['fake'])
        g_loss_msd = generator_loss(disc_outputs['msd']['fake'])
        losses['g_loss'] = g_loss_mpd + g_loss_msd
        
        # Feature matching losses
        fm_loss_mpd = feature_matching_loss(
            disc_outputs['mpd']['fmap_real'],
            disc_outputs['mpd']['fmap_fake']
        )
        fm_loss_msd = feature_matching_loss(
            disc_outputs['msd']['fmap_real'],
            disc_outputs['msd']['fmap_fake']
        )
        losses['fm_loss'] = (fm_loss_mpd + fm_loss_msd) * self.lambda_fm
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

    def compute_discriminator_loss(
        self,
        disc_outputs: dict,
    ) -> dict:
        """
        Compute discriminator losses.
        
        Args:
            disc_outputs: Dictionary of discriminator outputs
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # MPD loss
        d_loss_mpd = discriminator_loss(
            disc_outputs['mpd']['real'],
            disc_outputs['mpd']['fake']
        )
        losses['d_loss_mpd'] = d_loss_mpd
        
        # MSD loss
        d_loss_msd = discriminator_loss(
            disc_outputs['msd']['real'],
            disc_outputs['msd']['fake']
        )
        losses['d_loss_msd'] = d_loss_msd
        
        # Total loss
        losses['total'] = d_loss_mpd + d_loss_msd
        
        return losses


class SimpleLoss(nn.Module):
    """
    Simplified loss for lightweight discriminator.
    """
    
    def __init__(
        self,
        lambda_mel: float = 45.0,
        lambda_fm: float = 2.0,
    ):
        super().__init__()
        
        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm

    def compute_generator_loss(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        d_fake: torch.Tensor,
        fmap_real: List[torch.Tensor],
        fmap_fake: List[torch.Tensor],
        mel_coarse: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute generator losses.
        """
        losses = {}
        
        # Mel reconstruction loss
        mel_loss = F.l1_loss(mel_pred, mel_target)
        losses['mel_loss'] = mel_loss * self.lambda_mel
        
        # Coarse mel loss
        if mel_coarse is not None:
            coarse_loss = F.l1_loss(mel_coarse, mel_target)
            losses['coarse_loss'] = coarse_loss * self.lambda_mel * 0.5
        
        # Adversarial loss
        g_loss = torch.mean((1 - d_fake) ** 2)
        losses['g_loss'] = g_loss
        
        # Feature matching loss
        fm_loss = 0.0
        for r, g in zip(fmap_real, fmap_fake):
            fm_loss += F.l1_loss(g, r.detach())
        losses['fm_loss'] = fm_loss * self.lambda_fm
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

    def compute_discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> dict:
        """
        Compute discriminator losses.
        """
        losses = {}
        
        r_loss = torch.mean((1 - d_real) ** 2)
        g_loss = torch.mean(d_fake ** 2)
        losses['total'] = r_loss + g_loss
        
        return losses


# =============================================================================
# Knowledge Distillation Loss (from CFM teacher)
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss from CFM teacher.
    
    Helps the GAN decoder learn from the pre-trained CFM decoder.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        lambda_kd: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.lambda_kd = lambda_kd

    def forward(
        self,
        student_mel: torch.Tensor,
        teacher_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> dict:
        """
        Compute distillation loss.
        
        Args:
            student_mel: GAN decoder output
            teacher_mel: CFM decoder output (soft target)
            target_mel: Ground truth mel
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Hard target loss (ground truth)
        hard_loss = F.l1_loss(student_mel, target_mel)
        losses['hard_loss'] = hard_loss
        
        # Soft target loss (teacher)
        # Apply temperature softening
        soft_student = student_mel / self.temperature
        soft_teacher = teacher_mel / self.temperature
        soft_loss = F.mse_loss(soft_student, soft_teacher)
        losses['soft_loss'] = soft_loss * self.lambda_kd * (self.temperature ** 2)
        
        # Total loss
        losses['total'] = losses['hard_loss'] + losses['soft_loss']
        
        return losses


if __name__ == "__main__":
    # Test losses
    batch_size = 2
    n_mel = 80
    seq_len = 100
    
    mel_pred = torch.randn(batch_size, n_mel, seq_len)
    mel_target = torch.randn(batch_size, n_mel, seq_len)
    
    # Test multi-resolution loss
    print("Testing MelMultiResolutionLoss...")
    mr_loss = MelMultiResolutionLoss()
    loss = mr_loss(mel_pred, mel_target)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test distillation loss
    print("\nTesting DistillationLoss...")
    teacher_mel = torch.randn(batch_size, n_mel, seq_len)
    kd_loss = DistillationLoss()
    losses = kd_loss(mel_pred, teacher_mel, mel_target)
    print(f"  Hard loss: {losses['hard_loss'].item():.4f}")
    print(f"  Soft loss: {losses['soft_loss'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
