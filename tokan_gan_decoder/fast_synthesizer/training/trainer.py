"""
GAN Decoder Trainer with WandB Integration

Trains a single-forward GAN decoder to replace TokAN's iterative CFM decoder.

Key features:
- WandB logging with mel spectrogram visualization
- EMA for smoother convergence
- Proper AMP handling with single scaler.update() per step
- Per-epoch scheduler stepping
- Masked discriminator inputs
"""

import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from gan_decoder import get_gan_decoder
from discriminators import CombinedDiscriminator, LightweightDiscriminator
from losses import GANLoss, SimpleLoss
from gan_decoder_datamodule import GANDecoderDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * p.data
    
    def apply(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]
    
    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


# =============================================================================
# Trainer
# =============================================================================

class GANDecoderTrainer:
    """GAN Decoder Trainer with WandB integration."""
    
    def __init__(self, config: Dict, use_wandb: bool = True):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Set seed
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Setup components
        self._setup_models()
        self._setup_data()
        self._setup_optimizers()
        self._setup_losses()
        self._setup_amp()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # EMA
        ema_decay = config["training"].get("ema_decay", 0.999)
        self.ema = EMA(self.generator, decay=ema_decay)
        
        self._log_setup_info()

    def _setup_models(self):
        """Initialize generator and discriminator."""
        cfg = self.config["model"]
        disc_cfg = self.config["discriminator"]
        
        self.generator = get_gan_decoder(
            decoder_type=cfg["decoder_type"],
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            hidden_channels=cfg["hidden_channels"],
            kernel_sizes=tuple(cfg["kernel_sizes"]),
            dilations=tuple(tuple(d) for d in cfg["dilations"]),
            n_res_blocks=cfg["n_res_blocks"],
            spk_emb_dim=cfg["spk_emb_dim"],
            dropout=cfg["dropout"],
        ).to(self.device)
        
        if disc_cfg["type"] == "combined":
            self.discriminator = CombinedDiscriminator(
                periods=tuple(disc_cfg["periods"]),
                n_scales=disc_cfg["n_scales"],
                n_mel=disc_cfg["n_mel"],
            ).to(self.device)
            self.use_combined_disc = True
        else:
            self.discriminator = LightweightDiscriminator(n_mel=disc_cfg["n_mel"]).to(self.device)
            self.use_combined_disc = False

    def _setup_data(self):
        """Initialize data module."""
        cfg = self.config["data"]
        
        self.data_module = GANDecoderDataModule(
            data_dir=cfg["data_dir"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            test_speakers=cfg.get("test_speakers", ["EBVS", "SKA"]),
            sentences_per_speaker_test=cfg.get("sentences_per_speaker_test", 50),
            sentences_per_speaker_val=cfg.get("sentences_per_speaker_val", 50),
            max_mel_length=cfg.get("max_mel_length", 1000),
            h_y_dim=self.config["model"]["in_channels"],
            n_mels=self.config["model"]["out_channels"],
            seed=self.config.get("seed", 42),
        )
        self.data_module.setup("fit")
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()

    def _setup_optimizers(self):
        """Initialize optimizers and per-epoch schedulers."""
        cfg = self.config["training"]
        
        self.optimizer_g = AdamW(
            self.generator.parameters(),
            lr=cfg["lr_g"],
            betas=tuple(cfg["betas"]),
            weight_decay=cfg["weight_decay"],
        )
        self.optimizer_d = AdamW(
            self.discriminator.parameters(),
            lr=cfg["lr_d"],
            betas=tuple(cfg["betas"]),
            weight_decay=cfg["weight_decay"],
        )
        
        # Per-epoch schedulers
        total_epochs = cfg["max_epochs"]
        warmup_epochs = cfg.get("warmup_epochs", 5)
        min_lr = cfg.get("min_lr", 1e-6)
        
        def make_scheduler(optimizer):
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
            return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
        
        self.scheduler_g = make_scheduler(self.optimizer_g)
        self.scheduler_d = make_scheduler(self.optimizer_d)

    def _setup_losses(self):
        """Initialize loss functions."""
        cfg = self.config["training"]
        
        if self.use_combined_disc:
            self.loss_fn = GANLoss(
                lambda_mel=cfg["lambda_mel"],
                lambda_fm=cfg["lambda_fm"],
                lambda_mr=cfg.get("lambda_mr", 1.0),
            )
        else:
            self.loss_fn = SimpleLoss(
                lambda_mel=cfg["lambda_mel"],
                lambda_fm=cfg["lambda_fm"],
            )

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        precision = self.config.get("precision", "32")
        self.use_amp = "16" in precision or "bf16" in precision
        self.amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
        # Only use GradScaler for fp16, not bf16
        self.scaler = GradScaler() if self.use_amp and "16-mixed" == precision else None

    def _setup_logging(self):
        """Setup logging directories and wandb."""
        self.ckpt_dir = Path(self.config["checkpoint"]["save_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "gan-decoder-tokan"),
                name=self.config.get("wandb_run_name", f"gan_{datetime.now():%Y%m%d_%H%M%S}"),
                config=self.config,
            )

    def _log_setup_info(self):
        """Log setup information."""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        
        logger.info("=" * 60)
        logger.info("GAN DECODER TRAINER")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Precision: {self.config.get('precision', '32')}")
        logger.info(f"Generator params: {g_params:,}")
        logger.info(f"Discriminator params: {d_params:,}")
        logger.info(f"Train samples: {len(self.data_module.train_dataset)}")
        logger.info(f"Val samples: {len(self.data_module.val_dataset)}")
        logger.info(f"WandB: {'enabled' if self.use_wandb else 'disabled'}")
        logger.info("=" * 60)
        
        self.data_module.print_split_summary()

    def _compute_generator_loss(self, mel_pred, mel_target, mel_coarse, mask, use_disc):
        """Compute generator loss with proper masking."""
        cfg = self.config["training"]
        
        if not use_disc:
            # Warmup: reconstruction only
            loss = F.l1_loss(mel_pred * mask, mel_target * mask) * cfg["lambda_mel"]
            return {"total": loss, "mel_loss": loss}
        
        # Apply mask before discriminator
        mel_pred_m = mel_pred * mask
        mel_target_m = mel_target * mask
        
        if self.use_combined_disc:
            disc_out = self.discriminator(mel_target_m, mel_pred_m)
            return self._combined_disc_g_loss(mel_pred, mel_target, mel_coarse, mask, disc_out)
        else:
            _, d_fake, fmap_r, fmap_f = self.discriminator(mel_target_m, mel_pred_m)
            return self._simple_disc_g_loss(mel_pred, mel_target, mel_coarse, mask, d_fake, fmap_r, fmap_f)

    def _combined_disc_g_loss(self, mel_pred, mel_target, mel_coarse, mask, disc_out):
        """Generator loss for combined discriminator."""
        cfg = self.config["training"]
        losses = {}
        
        # Mel L1 loss
        losses["mel_loss"] = F.l1_loss(mel_pred * mask, mel_target * mask) * cfg["lambda_mel"]
        
        # Multi-resolution loss (from GANLoss class)
        if hasattr(self.loss_fn, 'multi_res_loss'):
            losses["mr_loss"] = self.loss_fn.multi_res_loss(mel_pred * mask, mel_target * mask) * cfg.get("lambda_mr", 1.0)
        else:
            losses["mr_loss"] = torch.tensor(0.0, device=mel_pred.device)
        
        # Coarse loss
        if mel_coarse is not None:
            losses["coarse_loss"] = F.l1_loss(mel_coarse * mask, mel_target * mask) * cfg["lambda_mel"] * 0.5
        
        # Adversarial loss
        g_loss = sum(torch.mean((1 - d) ** 2) for d in disc_out["mpd"]["fake"])
        g_loss += sum(torch.mean((1 - d) ** 2) for d in disc_out["msd"]["fake"])
        losses["g_loss"] = g_loss
        
        # Feature matching loss
        fm_loss = 0.0
        for r_list, f_list in [(disc_out["mpd"]["fmap_real"], disc_out["mpd"]["fmap_fake"]),
                               (disc_out["msd"]["fmap_real"], disc_out["msd"]["fmap_fake"])]:
            for r, f in zip(r_list, f_list):
                for r_feat, f_feat in zip(r, f):
                    fm_loss += F.l1_loss(f_feat, r_feat.detach())
        losses["fm_loss"] = fm_loss * cfg["lambda_fm"]
        
        losses["total"] = sum(losses.values())
        return losses

    def _simple_disc_g_loss(self, mel_pred, mel_target, mel_coarse, mask, d_fake, fmap_r, fmap_f):
        """Generator loss for simple discriminator."""
        cfg = self.config["training"]
        losses = {}
        
        losses["mel_loss"] = F.l1_loss(mel_pred * mask, mel_target * mask) * cfg["lambda_mel"]
        
        if mel_coarse is not None:
            losses["coarse_loss"] = F.l1_loss(mel_coarse * mask, mel_target * mask) * cfg["lambda_mel"] * 0.5
        
        losses["g_loss"] = torch.mean((1 - d_fake) ** 2)
        
        fm_loss = sum(F.l1_loss(f, r.detach()) for r, f in zip(fmap_r, fmap_f))
        losses["fm_loss"] = fm_loss * cfg["lambda_fm"]
        
        losses["total"] = sum(losses.values())
        return losses

    def _compute_discriminator_loss(self, mel_pred, mel_target, mask):
        """Compute discriminator loss."""
        mel_pred_m = mel_pred.detach() * mask
        mel_target_m = mel_target * mask
        
        if self.use_combined_disc:
            disc_out = self.discriminator(mel_target_m, mel_pred_m)
            d_loss = 0.0
            for key in ["mpd", "msd"]:
                for dr, df in zip(disc_out[key]["real"], disc_out[key]["fake"]):
                    d_loss += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
            return {"total": d_loss}
        else:
            d_real, d_fake, _, _ = self.discriminator(mel_target_m, mel_pred_m)
            d_loss = torch.mean((1 - d_real) ** 2) + torch.mean(d_fake ** 2)
            return {"total": d_loss}

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        cfg = self.config["training"]
        use_disc = self.current_epoch >= cfg.get("discriminator_start_epoch", 5)
        grad_clip = cfg.get("grad_clip", 5.0)
        log_interval = self.config["logging"]["log_every_n_steps"]
        
        metrics = {"g_total": 0.0, "g_mel": 0.0, "g_adv": 0.0, "d_total": 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            h_y = batch["h_y"].to(self.device)
            mel_target = batch["mel"].to(self.device)
            spk_embed = batch["spk_embed"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # === Generator ===
            self.optimizer_g.zero_grad()
            
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                mel_pred, mel_coarse = self.generator(h_y, mask, spk_embed)
                g_losses = self._compute_generator_loss(mel_pred, mel_target, mel_coarse, mask, use_disc)
            
            if self.scaler:
                self.scaler.scale(g_losses["total"]).backward()
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip)
                self.scaler.step(self.optimizer_g)
            else:
                g_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip)
                self.optimizer_g.step()
            
            self.ema.update()
            
            # === Discriminator ===
            d_loss_val = 0.0
            if use_disc:
                self.optimizer_d.zero_grad()
                
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    d_losses = self._compute_discriminator_loss(mel_pred, mel_target, mask)
                
                if self.scaler:
                    self.scaler.scale(d_losses["total"]).backward()
                    self.scaler.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), grad_clip)
                    self.scaler.step(self.optimizer_d)
                else:
                    d_losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), grad_clip)
                    self.optimizer_d.step()
                
                d_loss_val = d_losses["total"].item()
            
            # Update scaler once per step
            if self.scaler:
                self.scaler.update()
            
            # Accumulate metrics
            metrics["g_total"] += g_losses["total"].item()
            metrics["g_mel"] += g_losses.get("mel_loss", g_losses["total"]).item()
            metrics["g_adv"] += g_losses.get("g_loss", torch.tensor(0.0)).item() if isinstance(g_losses.get("g_loss"), torch.Tensor) else 0.0
            metrics["d_total"] += d_loss_val
            
            pbar.set_postfix(g=f"{g_losses['total'].item():.3f}", d=f"{d_loss_val:.3f}")
            
            # Logging
            if self.use_wandb and self.global_step % log_interval == 0:
                wandb.log({
                    "train/g_loss": g_losses["total"].item(),
                    "train/g_mel_loss": g_losses.get("mel_loss", g_losses["total"]).item(),
                    "train/d_loss": d_loss_val,
                    "train/lr_g": self.optimizer_g.param_groups[0]["lr"],
                    "global_step": self.global_step,
                })
            
            self.global_step += 1
        
        # Average metrics
        n = len(self.train_loader)
        return {k: v / n for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> Dict[str, float]:
        """Run validation."""
        if use_ema:
            self.ema.apply()
        self.generator.eval()
        
        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            h_y = batch["h_y"].to(self.device)
            mel_target = batch["mel"].to(self.device)
            spk_embed = batch["spk_embed"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            mel_pred, _ = self.generator(h_y, mask, spk_embed)
            total_loss += F.l1_loss(mel_pred * mask, mel_target * mask).item()
        
        if use_ema:
            self.ema.restore()
        
        return {"mel_loss": total_loss / len(self.val_loader)}

    @torch.no_grad()
    def test(self) -> Dict[str, Dict[str, float]]:
        """Run test evaluation."""
        self.ema.apply()
        self.generator.eval()
        self.data_module.setup("test")
        
        results = {}
        
        for name, loader in [("test_speaker", self.data_module.get_test_speaker_dataloader()),
                             ("test_sentence", self.data_module.get_test_sentence_dataloader())]:
            total_loss, n = 0.0, 0
            for batch in tqdm(loader, desc=f"Test ({name})"):
                h_y = batch["h_y"].to(self.device)
                mel_target = batch["mel"].to(self.device)
                spk_embed = batch["spk_embed"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                mel_pred, _ = self.generator(h_y, mask, spk_embed)
                total_loss += F.l1_loss(mel_pred * mask, mel_target * mask).item()
                n += 1
            
            results[name] = {"mel_loss": total_loss / max(n, 1)}
        
        self.ema.restore()
        return results

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        cfg = self.config["checkpoint"]
        
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "ema_shadow": self.ema.shadow,
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "scheduler_g_state_dict": self.scheduler_g.state_dict(),
            "scheduler_d_state_dict": self.scheduler_d.state_dict(),
            "config": self.config,
        }
        
        # Always save latest
        torch.save(ckpt, self.ckpt_dir / "latest.pt")
        
        # Save periodic checkpoint
        save_every = cfg.get("save_every_n_epochs", 10)
        if self.current_epoch % save_every == 0 and self.current_epoch > 0:
            torch.save(ckpt, self.ckpt_dir / f"epoch_{self.current_epoch}.pt")
            
            # Keep only last N checkpoints
            keep_n = cfg.get("keep_last_n", 5)
            ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
            for old in ckpts[:-keep_n]:
                old.unlink()
        
        # Save best
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best.pt")
            logger.info(f"✓ New best model (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        self.current_epoch = ckpt["epoch"] + 1  # Resume from next epoch
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        
        self.generator.load_state_dict(ckpt["generator_state_dict"])
        self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        
        if "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]
        
        self.optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d_state_dict"])
        
        logger.info(f"✓ Resumed from epoch {self.current_epoch - 1}")

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        cfg = self.config["training"]
        max_epochs = cfg["max_epochs"]
        
        logger.info(f"\nStarting training for {max_epochs} epochs...")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Step schedulers (per epoch!)
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Validate
            val_metrics = self.validate(use_ema=True)
            
            logger.info(
                f"Epoch {epoch}: g_loss={train_metrics['g_total']:.4f}, "
                f"d_loss={train_metrics['d_total']:.4f}, val_mel={val_metrics['mel_loss']:.4f}"
            )
            
            # Check best
            is_best = val_metrics["mel_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["mel_loss"]
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # WandB epoch logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "epoch/g_loss": train_metrics["g_total"],
                    "epoch/d_loss": train_metrics["d_total"],
                    "val/mel_loss": val_metrics["mel_loss"],
                    "val/best_mel_loss": self.best_val_loss,
                })
        
        # Final test
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)
        
        test_results = self.test()
        logger.info(f"Test (held-out speakers): {test_results['test_speaker']['mel_loss']:.4f}")
        logger.info(f"Test (unseen sentences): {test_results['test_sentence']['mel_loss']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                "test/speaker_mel_loss": test_results["test_speaker"]["mel_loss"],
                "test/sentence_mel_loss": test_results["test_sentence"]["mel_loss"],
            })
            wandb.finish()
        
        logger.info("\n✓ Training complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GAN Decoder")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    trainer = GANDecoderTrainer(config, use_wandb=not args.no_wandb)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()