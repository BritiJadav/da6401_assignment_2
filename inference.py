import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models import MultiTaskPerceptionModel
from losses import IoULoss


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="multitask_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def compute_iou(pred_boxes, target_boxes, eps=1e-6):
    """Compute IoU for evaluation (same logic as loss, but returns IoU)."""

    px, py, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx, ty, tw, th = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    p_x1 = px - pw / 2
    p_y1 = py - ph / 2
    p_x2 = px + pw / 2
    p_y2 = py + ph / 2

    t_x1 = tx - tw / 2
    t_y1 = ty - th / 2
    t_x2 = tx + tw / 2
    t_y2 = ty + th / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    pred_area = pw * ph
    target_area = tw * th

    union = pred_area + target_area - inter_area + eps
    iou = inter_area / union

    return iou


def compute_segmentation_accuracy(pred_mask, true_mask):
    """Pixel-wise accuracy."""
    correct = (pred_mask == true_mask).float().sum()
    total = torch.numel(true_mask)
    return correct / total


def evaluate():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    test_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="val"  # or "test" depending on dataset
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # ---- Model ----
    model = MultiTaskPerceptionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ---- Metrics ----
    cls_correct = 0
    cls_total = 0

    iou_scores = []
    seg_accs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)

            cls_logits = outputs["classification"]
            pred_boxes = outputs["localization"]
            seg_logits = outputs["segmentation"]

            # ---- Classification Accuracy ----
            preds = torch.argmax(cls_logits, dim=1)
            cls_correct += (preds == labels).sum().item()
            cls_total += labels.size(0)

            # ---- IoU ----
            iou = compute_iou(pred_boxes, bboxes)
            iou_scores.extend(iou.cpu().tolist())

            # ---- Segmentation Accuracy ----
            seg_preds = torch.argmax(seg_logits, dim=1)
            acc = compute_segmentation_accuracy(seg_preds, masks)
            seg_accs.append(acc.item())

    # ---- Final Metrics ----
    cls_acc = cls_correct / cls_total
    mean_iou = sum(iou_scores) / len(iou_scores)
    mean_seg_acc = sum(seg_accs) / len(seg_accs)

    print("\n===== Evaluation Results =====")
    print(f"Classification Accuracy: {cls_acc:.4f}")
    print(f"Mean IoU (Localization): {mean_iou:.4f}")
    print(f"Segmentation Pixel Accuracy: {mean_seg_acc:.4f}")


if __name__ == "__main__":
    evaluate()