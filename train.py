import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def get_args():
    parser = argparse.ArgumentParser(description="Train the multi-task perception model.")

    parser.add_argument("--data_dir",   type=str,   default=".",    help="Root directory of the Oxford-IIIT Pet dataset")
    parser.add_argument("--batch_size", type=int,   default=4,      help="Training batch size")
    parser.add_argument("--epochs",     type=int,   default=10,     help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--device",     type=str,   default="cpu",  help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--save_path",  type=str,   default="multitask_model.pth", help="Where to save the model")

    return parser.parse_args()


def train():
    print("Training started...")
    args = get_args()

    # ---- Device ----
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset ----
    train_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="train",
        image_size=224,
    )
    val_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="val",
        image_size=224,
    )
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # keep 0 for Windows compatibility
        pin_memory=use_cuda,    # faster CPU → GPU transfer when using CUDA
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
    ).to(device)
    print("Model created.")

    # ---- Losses ----
    cls_criterion = nn.CrossEntropyLoss()
    loc_criterion = IoULoss(reduction="mean")
    seg_criterion = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    for epoch in range(args.epochs):
        model.train()
        total_loss     = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        total_seg_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)

            # Forward
            outputs  = model(images)
            cls_out  = outputs["classification"]
            loc_out  = outputs["localization"]
            seg_out  = outputs["segmentation"]

            # Losses
            cls_loss = cls_criterion(cls_out, labels)
            loc_loss = loc_criterion(loc_out, bboxes)
            seg_loss = seg_criterion(seg_out, masks)
            loss     = cls_loss + loc_loss + seg_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss     += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()
            total_seg_loss += seg_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(cls={cls_loss.item():.3f}, "
                    f"loc={loc_loss.item():.3f}, "
                    f"seg={seg_loss.item():.3f})"
                )

        n = len(train_loader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Avg Loss: {total_loss/n:.4f} | "
            f"cls: {total_cls_loss/n:.4f} | "
            f"loc: {total_loc_loss/n:.4f} | "
            f"seg: {total_seg_loss/n:.4f}"
        )

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs  = model(images)
                cls_out  = outputs["classification"]
                preds    = torch.argmax(cls_out, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        print(f"  Val Classification Accuracy: {val_acc:.4f}")

    # ---- Save model ----
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    train()
