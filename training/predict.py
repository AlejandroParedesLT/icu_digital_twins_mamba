"""Run inference on a held-out dataset and compute evaluation metrics."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from odyssey.data.dataset import (
    FinetuneDataset,
    FinetuneDatasetDecoder,
    FinetuneMultiDataset,
)
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_bert.model import BertFinetune, BertPretrain
from odyssey.models.cehr_big_bird.model import BigBirdFinetune, BigBirdPretrain
from odyssey.models.ehr_mamba.model import MambaFinetune, MambaPretrain
from odyssey.models.model_utils import load_config, load_finetune_data
from odyssey.utils.utils import seed_everything

ADDITIONAL_TOKEN_TYPES = [
    "type_tokens",
    "age_tokens",
    "time_tokens",
    "position_tokens",
    "visit_tokens",
]


# ---------------------------------------------------------------------------
# Helpers (copied from finetune.py so this script is self-contained)
# ---------------------------------------------------------------------------

def extract_pretrained_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and normalize Lightning/DataParallel key prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        normalized[k] = value
    return normalized


def adapt_state_dict_to_model(
    model: torch.nn.Module,
    pretrained_state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], list[str]]:
    model_state = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    resized_keys: list[str] = []
    allowed_resizable = {
        "embeddings.word_embeddings.weight",
        "model.backbone.embeddings.weight",
        "model.lm_head.weight",
    }
    for key, value in pretrained_state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue
        if key in allowed_resizable and value.ndim == 2 and value.shape[1] == target.shape[1]:
            resized = target.clone()
            n = min(value.shape[0], target.shape[0])
            resized[:n] = value[:n]
            adapted[key] = resized
            resized_keys.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")
    return adapted, resized_keys


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Return the highest precision achievable at or above *target_recall*."""
    thresholds = np.unique(y_prob)
    best_precision = 0.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall:
            prec = precision_score(y_true, y_pred, zero_division=0)
            best_precision = max(best_precision, prec)
    return best_precision


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
    problem_type: str = "single_label_classification",
) -> Dict[str, float]:
    """Compute AUC-ROC, AUPRC, and Precision@Recall80."""
    metrics: Dict[str, float] = {}

    if problem_type == "single_label_classification":
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
        metrics["AUPRC"] = average_precision_score(y_true, y_prob)
        metrics[f"Precision@Rec{int(target_recall * 100)}"] = precision_at_recall(
            y_true, y_prob, target_recall
        )
    else:
        # Multi-label: macro average across labels
        n_labels = y_true.shape[1] if y_true.ndim > 1 else 1
        auc_list, auprc_list, prec_list = [], [], []
        for i in range(n_labels):
            col_true = y_true[:, i]
            col_prob = y_prob[:, i]
            if len(np.unique(col_true)) < 2:
                continue  # skip labels with no positive examples
            auc_list.append(roc_auc_score(col_true, col_prob))
            auprc_list.append(average_precision_score(col_true, col_prob))
            prec_list.append(precision_at_recall(col_true, col_prob, target_recall))
        metrics["AUC"] = float(np.mean(auc_list))
        metrics["AUPRC"] = float(np.mean(auprc_list))
        metrics[f"Precision@Rec{int(target_recall * 100)}"] = float(np.mean(prec_list))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace, pre_model_config: Dict[str, Any], fine_model_config: Dict[str, Any]) -> None:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load held-out test data
    # ------------------------------------------------------------------
    _, fine_test = load_finetune_data(
        args.data_dir,
        args.sequence_file,
        args.id_file,
        args.valid_scheme,
        args.num_finetune_patients,
    )

    if not args.is_multi_model:
        if args.label_name not in fine_test.columns:
            available = sorted(c for c in fine_test.columns if c.startswith("label_"))
            raise KeyError(
                f"Label column '{args.label_name}' not found. Available: {available}"
            )
        if not args.is_decoder:
            fine_test.rename(columns={args.label_name: "label"}, inplace=True)

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
    tokenizer.fit_on_vocab(with_tasks=bool(args.tasks))

    # ------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # ------------------------------------------------------------------
    if args.is_decoder:
        test_dataset = FinetuneDatasetDecoder(
            data=fine_test,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=None,
            max_len=args.max_len,
            is_single_head=not fine_model_config.get("multi_head", False),
            additional_token_types=ADDITIONAL_TOKEN_TYPES,
            task_to_index_map={task: idx for idx, task in enumerate(args.tasks)},
        )
    elif args.is_multi_model:
        test_dataset = FinetuneMultiDataset(
            data=fine_test,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=None,
            max_len=args.max_len,
        )
    else:
        test_dataset = FinetuneDataset(
            data=fine_test,
            tokenizer=tokenizer,
            max_len=args.max_len,
            additional_token_types=ADDITIONAL_TOKEN_TYPES,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # 4. Build model architecture (same as finetune.py)
    # ------------------------------------------------------------------
    if args.model_type == "cehr_bert":
        pretrained_model = BertPretrain(
            args=args,
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **pre_model_config,
        )
        model = BertFinetune(args=args, pretrained_model=pretrained_model, **fine_model_config)

    elif args.model_type in ("cehr_bigbird", "cehr_multibird"):
        pretrained_model = BigBirdPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **pre_model_config,
        )
        model = BigBirdFinetune(
            pretrained_model=pretrained_model,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
            **fine_model_config,
        )

    elif args.model_type == "ehr_mamba":
        if fine_model_config.get("multi_head", False):
            fine_model_config["num_tasks"] = len(args.tasks)
        pretrained_model = MambaPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
            **pre_model_config,
        )
        model = MambaFinetune(
            pretrained_model=pretrained_model,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
            **fine_model_config,
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # ------------------------------------------------------------------
    # 5. Load finetuned checkpoint weights into the model
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning wraps the module under "model." — strip that prefix if present
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 6. Inference loop
    # ------------------------------------------------------------------
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            # Move every tensor in the batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch.get("labels", batch.get("label"))

            outputs = model(**{k: v for k, v in batch.items() if k not in ("labels", "label")})

            # Support both plain tensor outputs and HuggingFace-style objects
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if args.problem_type == "single_label_classification":
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1))
                elif logits.shape[-1] == 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # ------------------------------------------------------------------
    # 7. Save raw predictions
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_path = os.path.join(args.output_dir, "predictions.npz")
    np.savez(predictions_path, y_prob=y_prob, y_true=y_true)
    print(f"\nPredictions saved to: {predictions_path}")

    # ------------------------------------------------------------------
    # 8. Compute & print metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(
        y_true=y_true,
        y_prob=y_prob,
        target_recall=args.target_recall,
        problem_type=args.problem_type,
    )

    print("\n========== Evaluation Metrics ==========")
    for name, value in metrics.items():
        print(f"  {name:<30} {value:.4f}")
    print("========================================\n")

    # Optionally persist metrics as JSON
    import json
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on held-out data and compute metrics.")

    # Model
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["cehr_bert", "cehr_bigbird", "ehr_mamba", "cehr_multibird"])
    parser.add_argument("--config-dir", type=str, required=True, help="Path to model config directory")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to the finetuned .ckpt checkpoint to evaluate")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sequence-file", type=str, required=True)
    parser.add_argument("--id-file", type=str, required=True)
    parser.add_argument("--vocab-dir", type=str, required=True)
    parser.add_argument("--label-name", type=str, default=None)
    parser.add_argument("--valid_scheme", type=str, default="few_shot")
    parser.add_argument("--num_finetune_patients", type=str, required=True)
    parser.add_argument("--problem_type", type=str, required=True,
                        choices=["single_label_classification", "multi_label_classification"])
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--is-multi-model", type=bool, default=False)
    parser.add_argument("--is-decoder", type=bool, default=False)

    # Inference
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-recall", type=float, default=0.80,
                        help="Target recall for Precision@Recall metric (default: 0.80)")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save predictions.npz and metrics.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = load_config(args.config_dir, args.model_type)
    finetune_config = config["finetune"]
    for key, value in finetune_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    pre_model_config = config["model"]
    args.max_len = pre_model_config["max_seq_length"]

    args.tasks = args.tasks.strip().split(" ") if args.tasks else []

    fine_model_config = config["model_finetune"]
    main(args, pre_model_config, fine_model_config)