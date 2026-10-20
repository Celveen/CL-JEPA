import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")
    coeffs_cache = {}  # (T, device) -> coeffs; a host-to-device copy cannot run inside CUDA graph capture

    def objective_fn_last(z_obs_pred, z_obs_tgt, step=None):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, step=None, coeffs=None, base=base):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        if coeffs is None:
            key = (z_obs_pred["visual"].shape[1], z_obs_pred["visual"].device)
            if key not in coeffs_cache:
                c = np.array([base**i for i in range(key[0])], dtype=np.float32)
                coeffs_cache[key] = torch.tensor(c / np.sum(c)).to(key[1])
            coeffs = coeffs_cache[key]
        else:
            coeffs = coeffs.to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_staged(z_obs_pred, z_obs_tgt, step=None):
        if step is None:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 1: optimize only terminal match
        if step < z_obs_pred["visual"].shape[1] - 1:
            return objective_fn_last(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 2: use the full-horizon weighted objective
        else:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt, coeffs=None)

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "staged":
        return objective_fn_staged
    else:
        raise NotImplementedError
