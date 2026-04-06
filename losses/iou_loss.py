import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] predicted boxes (x_center, y_center, width, height)
            target_boxes: [B, 4] ground truth boxes (x_center, y_center, width, height)
        """

        # ---- Convert (x_center, y_center, w, h) → (x1, y1, x2, y2) ----

        # Pred boxes
        px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        p_x1 = px - pw / 2
        p_y1 = py - ph / 2
        p_x2 = px + pw / 2
        p_y2 = py + ph / 2

        # Target boxes
        tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        t_x1 = tx - tw / 2
        t_y1 = ty - th / 2
        t_x2 = tx + tw / 2
        t_y2 = ty + th / 2

        # ---- Intersection ----
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # ---- Areas ----
        pred_area = torch.clamp(pw, min=0) * torch.clamp(ph, min=0)
        target_area = torch.clamp(tw, min=0) * torch.clamp(th, min=0)

        # ---- Union ----
        union_area = pred_area + target_area - inter_area + self.eps

        # ---- IoU ----
        iou = inter_area / union_area

        # ---- Loss ----
        loss = 1 - iou  # IoU loss

        # ---- Reduction ----
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss