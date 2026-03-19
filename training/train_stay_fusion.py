"""Train stay-level multimodal fusion models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odyssey.data.stay_fusion_dataset import StayFusionDataset, stay_fusion_collate
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.ehr_mamba.model import MambaPretrain
from odyssey.models.fusion import (
    CrossAttentionFusionModel,
    GatedFusionModel,
    LateFusionModel,
    StayEHRMambaEncoder,
    StayImageEncoder,
    StayNCDEEncoder,
)
from odyssey.models.model_utils import load_config
from odyssey.utils.utils import seed_everything


def build_ehr_encoder(
    args: argparse.Namespace,
    tokenizer: ConceptTokenizer,
) -> StayEHRMambaEncoder:
    """Build either a pretrained Mamba EHR encoder or a fallback embedding encoder."""
    if args.ehr_checkpoint is None:
        return StayEHRMambaEncoder(
            pretrained_model=None,
            hidden_size=args.ehr_hidden_size,
            vocab_size=tokenizer.get_vocab_size(),
        )

    config = load_config(args.ehr_config_dir, "ehr_mamba")
    model = MambaPretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
        cls_idx=tokenizer.get_class_token_id(),
        **config["model"],
    )
    state = torch.load(args.ehr_checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return StayEHRMambaEncoder(
        pretrained_model=model,
        hidden_size=config["model"]["embedding_size"],
    )


def build_model(
    args: argparse.Namespace,
    tokenizer: ConceptTokenizer,
) -> nn.Module:
    """Construct the requested fusion architecture."""
    ehr_encoder = build_ehr_encoder(args, tokenizer)
    cde_encoder = StayNCDEEncoder(hidden_size=args.cde_hidden_size)
    image_encoder = StayImageEncoder(hidden_size=args.image_hidden_size)

    if args.model_type == "late":
        return LateFusionModel(
            ehr_encoder=ehr_encoder,
            cde_encoder=cde_encoder,
            image_encoder=image_encoder,
            ehr_dim=args.ehr_hidden_size,
            cde_dim=args.cde_hidden_size,
            image_dim=args.image_hidden_size,
            fusion_dim=args.fusion_dim,
            num_tasks=args.num_tasks,
            dropout=args.dropout,
        )
    if args.model_type == "gated":
        return GatedFusionModel(
            ehr_encoder=ehr_encoder,
            cde_encoder=cde_encoder,
            image_encoder=image_encoder,
            ehr_dim=args.ehr_hidden_size,
            cde_dim=args.cde_hidden_size,
            image_dim=args.image_hidden_size,
            fusion_dim=args.fusion_dim,
            num_tasks=args.num_tasks,
            dropout=args.dropout,
        )
    return CrossAttentionFusionModel(
        ehr_encoder=ehr_encoder,
        cde_encoder=cde_encoder,
        image_encoder=image_encoder,
        ehr_dim=args.ehr_hidden_size,
        cde_dim=args.cde_hidden_size,
        image_dim=args.image_hidden_size,
        fusion_dim=args.fusion_dim,
        num_tasks=args.num_tasks,
        num_layers=args.fusion_layers,
        num_heads=args.fusion_heads,
        dropout=args.dropout,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on the validation split."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            ehr = {key: value.to(device) for key, value in batch["ehr"].items()}
            cde_coeffs = batch["cde_coeffs"].to(device)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            modality_mask = {
                key: value.to(device) for key, value in batch["modality_mask"].items()
            }

            outputs = model(ehr, cde_coeffs, images, modality_mask)
            loss = criterion(outputs["logits"], labels)
            total_loss += loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

    return total_loss / max(total_examples, 1)


def train(args: argparse.Namespace) -> None:
    """Train the requested fusion model."""
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
    tokenizer.fit_on_vocab()

    dataset = StayFusionDataset(
        stay_sequences_path=args.stay_sequences_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        cde_coeffs_path=args.cde_coeffs_path,
        cde_meta_path=args.cde_meta_path,
        image_index_path=args.image_index_path,
        image_root=args.image_root,
        image_size=args.image_size,
        max_images_per_stay=args.max_images_per_stay,
    )

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=stay_fusion_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stay_fusion_collate,
    )

    model = build_model(args, tokenizer).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            ehr = {key: value.to(device) for key, value in batch["ehr"].items()}
            cde_coeffs = batch["cde_coeffs"].to(device)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            modality_mask = {
                key: value.to(device) for key, value in batch["modality_mask"].items()
            }

            outputs = model(ehr, cde_coeffs, images, modality_mask)
            loss = criterion(outputs["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

        train_loss = total_loss / max(total_examples, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    print(f"Saved checkpoints to {output_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stay-sequences-path", type=str, required=True)
    parser.add_argument("--vocab-dir", type=str, required=True)
    parser.add_argument("--cde-coeffs-path", type=str, default=None)
    parser.add_argument("--cde-meta-path", type=str, default=None)
    parser.add_argument("--image-index-path", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-type", type=str, choices=["late", "gated", "cross"], default="late")
    parser.add_argument("--ehr-checkpoint", type=str, default=None)
    parser.add_argument("--ehr-config-dir", type=str, default="odyssey/models/configs")

    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images-per-stay", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--ehr-hidden-size", type=int, default=768)
    parser.add_argument("--cde-hidden-size", type=int, default=32)
    parser.add_argument("--image-hidden-size", type=int, default=768)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
