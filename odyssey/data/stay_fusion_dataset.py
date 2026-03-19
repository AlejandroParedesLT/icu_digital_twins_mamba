"""Datasets for stay-level multimodal fusion training."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from odyssey.data.tokenizer import ConceptTokenizer


LABEL_COLUMNS = ("label_in_hosp_mortality", "label_mortality_28d", "label_sepsis")


class StayFusionDataset(Dataset):
    """Stay-level dataset for EHR, CDE, and image fusion."""

    def __init__(
        self,
        stay_sequences_path: str,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        cde_coeffs_path: Optional[str] = None,
        cde_meta_path: Optional[str] = None,
        image_index_path: Optional[str] = None,
        image_root: Optional[str] = None,
        split_path: Optional[str] = None,
        split_name: Optional[str] = None,
        image_transform: Optional[Any] = None,
        image_size: int = 224,
        max_images_per_stay: Optional[int] = None,
    ) -> None:
        self.sequences = pd.read_parquet(stay_sequences_path).copy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_root = Path(image_root) if image_root else None
        self.image_transform = image_transform
        self.image_size = image_size
        self.max_images_per_stay = max_images_per_stay

        if split_path and split_name:
            with open(split_path, "rb") as file:
                split_dict = pickle.load(file)
            allowed_ids = set(split_dict[split_name])
            self.sequences = self.sequences[
                self.sequences["stay_id"].astype(int).isin(allowed_ids)
            ].copy()

        self.sequences = self.sequences.sort_values("stay_id").reset_index(drop=True)
        self.ehr_rows = {
            int(row.stay_id): row for row in self.sequences.itertuples(index=False)
        }
        self.stay_ids = list(self.ehr_rows.keys())

        self.cde_coeffs = None
        self.cde_by_stay: Dict[int, torch.Tensor] = {}
        if cde_coeffs_path and cde_meta_path:
            coeffs = torch.load(cde_coeffs_path, map_location="cpu")
            meta = pd.read_parquet(cde_meta_path)
            self.cde_coeffs = coeffs
            for idx, stay_id in enumerate(meta["stay_id"].astype(int).tolist()):
                self.cde_by_stay[stay_id] = coeffs[idx]

        self.images_by_stay: Dict[int, List[Dict[str, str]]] = {}
        if image_index_path:
            image_index = pd.read_parquet(image_index_path)
            image_index = image_index.sort_values(["stay_id", "StudyDateTime", "study_id"])
            for stay_id, group in image_index.groupby("stay_id"):
                image_records: List[Dict[str, str]] = []
                for row in group.itertuples(index=False):
                    record: Dict[str, str] = {}
                    tensor_path = getattr(row, "tensor_path", None)
                    jpg_path = getattr(row, "jpg_path", None)
                    if isinstance(tensor_path, str) and tensor_path:
                        record["tensor_path"] = tensor_path
                    if isinstance(jpg_path, str) and jpg_path:
                        record["jpg_path"] = jpg_path
                    if record:
                        image_records.append(record)
                if self.max_images_per_stay is not None:
                    image_records = image_records[: self.max_images_per_stay]
                self.images_by_stay[int(stay_id)] = image_records

    def __len__(self) -> int:
        """Return the number of stay-aligned samples."""
        return len(self.stay_ids)

    def _tokenize_sequence(self, row: Any) -> Dict[str, torch.Tensor]:
        tokens = list(getattr(row, f"event_tokens_{self.max_len}"))
        encoded = self.tokenizer(
            " ".join(tokens),
            max_length=self.max_len,
        )
        return {
            "concept_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "type_ids": torch.tensor(list(getattr(row, f"type_tokens_{self.max_len}")), dtype=torch.long),
            "ages": torch.tensor(list(getattr(row, f"age_tokens_{self.max_len}")), dtype=torch.long),
            "time_stamps": torch.tensor(
                list(getattr(row, f"elapsed_tokens_{self.max_len}")),
                dtype=torch.float32,
            ),
            "visit_orders": torch.tensor(
                list(getattr(row, f"position_tokens_{self.max_len}")),
                dtype=torch.long,
            ),
            "visit_segments": torch.tensor(
                list(getattr(row, f"visit_tokens_{self.max_len}")),
                dtype=torch.long,
            ),
        }

    def _resolve_image_path(self, path_value: str) -> Path:
        resolved = Path(path_value)
        if self.image_root and not resolved.is_absolute():
            resolved = self.image_root / resolved
        return resolved

    def _default_image_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype=np.uint8)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return tensor

    def _load_images(self, stay_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_records = self.images_by_stay.get(stay_id, [])
        if not image_records:
            return torch.zeros(1, 3, self.image_size, self.image_size), torch.zeros(1, dtype=torch.float32)

        tensors = []
        for image_record in image_records:
            tensor_path = image_record.get("tensor_path")
            if tensor_path:
                resolved_tensor = self._resolve_image_path(tensor_path)
                if resolved_tensor.exists():
                    tensor = torch.load(resolved_tensor, map_location="cpu")
                    if tensor.dtype == torch.uint8:
                        tensor = tensor.float() / 255.0
                    tensors.append(tensor)
                    continue

            jpg_path = image_record.get("jpg_path")
            if not jpg_path:
                continue
            resolved_jpg = self._resolve_image_path(jpg_path)
            if not resolved_jpg.exists():
                continue
            image = Image.open(resolved_jpg).convert("RGB")
            if self.image_transform is not None:
                tensor = self.image_transform(image)
            else:
                tensor = self._default_image_tensor(image)
            tensors.append(tensor)

        if not tensors:
            return torch.zeros(1, 3, self.image_size, self.image_size), torch.zeros(1, dtype=torch.float32)

        return torch.stack(tensors), torch.ones(1, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one stay-aligned multimodal sample."""
        stay_id = self.stay_ids[index]
        row = self.ehr_rows[stay_id]
        ehr = self._tokenize_sequence(row)

        labels = torch.tensor(
            [
                int(getattr(row, "label_in_hosp_mortality", 0)),
                int(getattr(row, "label_mortality_28d", 0)),
                int(getattr(row, "label_sepsis", 0)),
            ],
            dtype=torch.float32,
        )

        cde_coeffs = self.cde_by_stay.get(stay_id)
        has_cde = torch.ones(1, dtype=torch.float32)
        if cde_coeffs is None:
            cde_coeffs = torch.zeros(1, 2)
            has_cde = torch.zeros(1, dtype=torch.float32)

        images, has_img = self._load_images(stay_id)

        return {
            "stay_id": stay_id,
            "ehr": ehr,
            "cde_coeffs": cde_coeffs,
            "images": images,
            "labels": labels,
            "modality_mask": {
                "ehr": torch.ones(1, dtype=torch.float32),
                "cde": has_cde,
                "img": has_img,
            },
        }


def stay_fusion_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate stay-level multimodal samples into one batch."""
    ehr_keys = batch[0]["ehr"].keys()
    ehr = {
        key: torch.stack([item["ehr"][key] for item in batch], dim=0)
        for key in ehr_keys
    }
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    stay_ids = torch.tensor([item["stay_id"] for item in batch], dtype=torch.long)

    cde_lengths = [item["cde_coeffs"].shape[0] for item in batch]
    max_cde_len = max(cde_lengths)
    cde_feature_dim = batch[0]["cde_coeffs"].shape[-1]
    cde_coeffs = torch.zeros(len(batch), max_cde_len, cde_feature_dim)
    for idx, item in enumerate(batch):
        coeffs = item["cde_coeffs"]
        cde_coeffs[idx, : coeffs.shape[0]] = coeffs

    image_counts = [item["images"].shape[0] for item in batch]
    max_images = max(image_counts)
    channels, height, width = batch[0]["images"].shape[1:]
    images = torch.zeros(len(batch), max_images, channels, height, width)
    for idx, item in enumerate(batch):
        current_images = item["images"]
        images[idx, : current_images.shape[0]] = current_images

    modality_mask = {
        modality: torch.stack(
            [item["modality_mask"][modality] for item in batch],
            dim=0,
        )
        for modality in ("ehr", "cde", "img")
    }

    return {
        "stay_ids": stay_ids,
        "ehr": ehr,
        "cde_coeffs": cde_coeffs,
        "images": images,
        "labels": labels,
        "modality_mask": modality_mask,
    }
