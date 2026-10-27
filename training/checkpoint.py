# =============================================================================
# training/checkpoint.py
# CheckpointManager: save and load training state for teacher and student.
#
# Teacher checkpoint contains:
#   epoch, model_state_dict, optimizer_state_dict, scaler_state_dict,
#   ema_state_dict, tirt_state_dict, scheduler_state_dict, train/val histories.
#
# Student checkpoint additionally contains:
#   L_im_hist, L_un_hist, L_an_hist (per-component loss histories).
#
# keep_n controls how many recent checkpoints to retain on disk
# (older ones are deleted automatically after each save).
# =============================================================================

import os
import glob
import torch


class CheckpointManager:
    """
    Saves checkpoints under `save_dir` as:
        ckpt_epoch_{epoch:04d}.pth

    Args:
        save_dir  Directory to write checkpoint files.
        keep_n    Number of most-recent checkpoints to retain (default 2).
    """

    def __init__(self, save_dir: str, keep_n: int = 2):
        self.save_dir = save_dir
        self.keep_n   = keep_n
        os.makedirs(save_dir, exist_ok=True)

    def save(self, bundle: dict, epoch: int) -> str:
        """
        Write checkpoint and prune old files.

        Args:
            bundle  Dict of state dicts and history lists to save.
            epoch   Current epoch number (used in filename).

        Returns:
            Path of the saved checkpoint file.
        """
        path = os.path.join(self.save_dir, f"ckpt_epoch_{epoch:04d}.pth")
        torch.save(bundle, path)
        print(f"   Saved checkpoint: {path}")

        # Prune old checkpoints
        all_ckpts = sorted(glob.glob(
            os.path.join(self.save_dir, "ckpt_epoch_*.pth")
        ))
        for old in all_ckpts[:-self.keep_n]:
            os.remove(old)
            print(f"   Removed old checkpoint: {old}")

        return path

    def load(self, path: str, device: str = "cpu") -> dict:
        """
        Load a checkpoint from disk.

        Args:
            path    Path to the .pth file.
            device  Target device for tensors.

        Returns:
            Dict of saved state dicts and histories.
        """
        print(f"Loading checkpoint: {path}")
        return torch.load(path, map_location=device, weights_only=False)

    def latest(self) -> str:
        """
        Return the path of the most recent checkpoint, or None if none exist.
        """
        ckpts = sorted(glob.glob(
            os.path.join(self.save_dir, "ckpt_epoch_*.pth")
        ))
        return ckpts[-1] if ckpts else None
