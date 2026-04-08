import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def get_args():
    parser = argparse.ArgumentParser(description="Train the multi-task perception model.")

    parser.add_argument("--data_dir",   type=str,   default=".",    help="Root directory of the Oxford-IIIT Pet dataset")
    parser.add_argument("--batch_size", type=int,   default=4,      help="Training batch size")
    parser.add_argument("--epochs",     type=int,   default=30,     help="Number of training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--device",     type=str,   default="cpu",  help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--save_path",  type=str,   default="multitask_model.pth", help="Where to save the model")
    parser.add_argument("--run_name",   type=str,   default="multitask-run-1",     help="W&B run name")
    parser.add_argument("--dropout_p",  type=float, default=0.5,    help="Dropout probability")

    return parser.parse_args()


def train():
    print("Training started...")
    args = get_args()

    # ---- Device ----
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # ---- W&B Initialization ----
    wandb.init(
        project="da6401-assignment-2",
        name=args.run_name,
        config={
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "learning_rate": args.lr,
            "device":        args.device,
        }
    )

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
        num_workers=0,
        pin_memory=use_cuda,
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
        dropout_p=args.dropout_p,
        load_ckpts=False,
    ).to(device)
    print("Model created.")

    # ---- Losses ----
    cls_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction="mean")
    seg_criterion = nn.CrossEntropyLoss()

    LAMBDA_MSE = 0.01   # scale down MSE to match IoU's [0,1] range

    # ---- Optimizer ----
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": 1e-4},
        {"params": model.encoder.parameters(),    "lr": 1e-4},
        {"params": model.localizer.parameters(),  "lr": 1e-3},
        {"params": model.dec4.parameters(),       "lr": 1e-3},
        {"params": model.dec3.parameters(),       "lr": 1e-3},
        {"params": model.dec2.parameters(),       "lr": 1e-3},
        {"params": model.dec1.parameters(),       "lr": 1e-3},
        {"params": model.dec0.parameters(),       "lr": 1e-3},
        {"params": model.seg_head.parameters(),   "lr": 1e-3},
    ])

    # ---- Scheduler ----
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ---- Training Loop ----
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
            loc_loss = LAMBDA_MSE * mse_criterion(loc_out, bboxes) + iou_criterion(loc_out, bboxes)
            seg_loss = seg_criterion(seg_out, masks)
            loss     = cls_loss + 10.0 * loc_loss + 5.0 * seg_loss

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
        avg_total = total_loss     / n
        avg_cls   = total_cls_loss / n
        avg_loc   = total_loc_loss / n
        avg_seg   = total_seg_loss / n

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Avg Loss: {avg_total:.4f} | "
            f"cls: {avg_cls:.4f} | "
            f"loc: {avg_loc:.4f} | "
            f"seg: {avg_seg:.4f}"
        )

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss    = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks  = batch["mask"].to(device)

                outputs  = model(images)
                cls_out  = outputs["classification"]
                preds    = torch.argmax(cls_out, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

                cls_loss = cls_criterion(outputs["classification"], labels)
                loc_loss = (LAMBDA_MSE * mse_criterion(outputs["localization"], bboxes)
                          + iou_criterion(outputs["localization"], bboxes))
                seg_loss = seg_criterion(outputs["segmentation"], masks)
                val_loss += (cls_loss + 10.0 * loc_loss + 5.0 * seg_loss).item()

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        print(f"  Val Classification Accuracy: {val_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # ---- W&B Logging ----
        wandb.log({
            "epoch":            epoch + 1,
            "train/total_loss": avg_total,
            "train/cls_loss":   avg_cls,
            "train/loc_loss":   avg_loc,
            "train/seg_loss":   avg_seg,
            "val/total_loss":   val_loss,
            "val/accuracy":     val_acc,
            "learning_rate":    scheduler.get_last_lr()[0],
        })

    # ---- Save Model ----
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    wandb.finish()


if __name__ == "__main__":
    train()
