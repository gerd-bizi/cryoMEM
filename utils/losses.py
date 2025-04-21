import torch
import numpy as np

def compute_amortized_loss(data_cuda, out_dict):
    """
    Computes the amortized loss as the average minimum head loss plus a penalty on psi and theta differences.

    Parameters:
        data_cuda: dictionary containing input tensors; must include key 'fproj' (B x 1 x H x W)
        out_dict: dictionary from the model containing keys 'pred_fproj', 'mask', and 'delta_angles'.
        psi_penalty_weight: weight for the psi penalty term.
        theta_penalty_weight: weight for the theta penalty term.

    Returns:
        loss: the computed loss value
    """
    # Reconstruction loss: Calculate the per-head loss
    fproj_input = data_cuda['fproj']  # shape: [B, 1, H, W]
    fproj_pred = out_dict['pred_fproj']  # shape: [B, M, H, W]
    mask = out_dict['mask']              # shape: [B, M, H, W]
    batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum(dim=(-1, -2))) / (mask.sum(dim=(-1, -2)) + 1e-8)  # shape: [B, M]

    return batch_head_loss

def temperature_schedule(epoch, 
                         hold_epochs=4, 
                         total_epochs=10,
                         T_start=3, 
                         T_end=0.3):
    """
    Piecewise temperature schedule:
    - Holds at T_start for `hold_epochs`
    - Anneals exponentially to T_end by `total_epochs`
    """
    if epoch < hold_epochs:
        return T_start
    else:
        decay_epochs = total_epochs - hold_epochs
        decay_factor = (T_end / T_start) ** (1 / decay_epochs)
        anneal_epoch = epoch - hold_epochs
        return T_start * (decay_factor ** anneal_epoch)

@torch.jit.script
def soft_head_loss(
    fproj_pred,    # [B, M, H, W]
    fproj_true,    # [B, 1, H, W]
    mask,          # [B, M, H, W]
    temperature: float = 0.1,
    ):
    B, M, H, W = fproj_pred.shape
    fproj_true = fproj_true.expand(-1, M, -1, -1)

    # 1) compute raw per‐head loss [B, M]
    per_head_loss = (torch.abs(fproj_pred - fproj_true) * mask) \
                        .sum(dim=(-1, -2)) \
                        / (mask.sum(dim=(-1, -2)) + 1e-8)

    # 2) softmin weights [B, M]
    weights = torch.softmax(-per_head_loss / temperature, dim=1)

    # 3) per‐sample weighted loss [B]
    loss_per_sample = (weights * per_head_loss).sum(dim=1)

    # 4) scalar loss
    loss = loss_per_sample.mean()

    return loss, per_head_loss, weights, loss_per_sample