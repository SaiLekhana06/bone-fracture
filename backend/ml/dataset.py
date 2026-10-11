import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

class SPECTDataset(Dataset):
    """
    PyTorch Dataset for SPECT bone scintigraphy images based on a CSV manifest.
    Columns required: image_path, patient_id, label, split (train / val / test)
    """

    def __init__(
        self,
        manifest_path: Path,
        split: Optional[str] = "train",
        transform: Optional[transforms.Compose] = None,
        base_dir: Optional[Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.parent
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []

        self._load_and_filter()

    def _load_and_filter(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"image_path", "patient_id", "label"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest must contain columns {required_cols}. Found: {reader.fieldnames}")

            for row in reader:
                # Filter by split if column present
                row_split = row.get("split", "").strip().lower()
                if self.split and row_split and row_split != self.split.lower():
                    continue

                img_path = Path(row["image_path"].strip())
                if not img_path.is_absolute():
                    img_path = self.base_dir / img_path

                label_val = int(row["label"].strip())
                self.samples.append({
                    "image_path": str(img_path),
                    "patient_id": row["patient_id"].strip(),
                    "label": label_val,
                    "split": row_split,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        label = sample["label"]
        patient_id = sample["patient_id"]

        try:
            with Image.open(img_path) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                if self.transform:
                    image_tensor = self.transform(pil_img)
                else:
                    image_tensor = transforms.ToTensor()(pil_img)
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {str(e)}")

        return image_tensor, label, patient_id, img_path


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Validates CSV manifest integrity, clinical class balance, patient overlap across splits.
    Guarantees no data leakage between patient cohorts.
    """
    if not manifest_path.exists():
        return {
            "valid": False,
            "error": f"Manifest file not found: {manifest_path}",
        }

    samples: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"image_path", "patient_id", "label"}
        if not required.issubset(set(fields)):
            return {
                "valid": False,
                "error": f"Missing required columns. Must include: {list(required)}. Found: {fields}",
            }
        for row in reader:
            samples.append(row)

    if len(samples) == 0:
        return {"valid": False, "error": "Manifest CSV contains no data rows."}

    # Groupings & patient tracking
    split_patients: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    class_counts: Dict[str, int] = {}
    missing_files: List[str] = []

    for row in samples:
        split = row.get("split", "unassigned").strip().lower()
        patient = row.get("patient_id", "").strip()
        label = row.get("label", "").strip()
        img_path = row.get("image_path", "").strip()

        # Class counts
        class_counts[label] = class_counts.get(label, 0) + 1

        # Check file existence if path provided
        resolved = Path(img_path)
        if not resolved.is_absolute():
            resolved = manifest_path.parent / resolved
        if not resolved.exists() and len(missing_files) < 10:
            missing_files.append(str(img_path))

        if split in split_patients:
            split_patients[split].add(patient)
            split_counts[split] += 1
        else:
            split_counts["unassigned"] += 1

    # Patient leakage check
    leak_train_val = split_patients["train"].intersection(split_patients["val"])
    leak_train_test = split_patients["train"].intersection(split_patients["test"])
    leak_val_test = split_patients["val"].intersection(split_patients["test"])

    leakages = []
    if leak_train_val:
        leakages.append(f"Leakage between train and val: {len(leak_train_val)} overlapping patients")
    if leak_train_test:
        leakages.append(f"Leakage between train and test: {len(leak_train_test)} overlapping patients")
    if leak_val_test:
        leakages.append(f"Leakage between val and test: {len(leak_val_test)} overlapping patients")

    has_leakage = len(leakages) > 0

    return {
        "valid": not has_leakage and len(missing_files) == 0,
        "total_samples": len(samples),
        "total_patients": len(set(r.get("patient_id", "") for r in samples)),
        "class_distribution": class_counts,
        "split_counts": split_counts,
        "split_patient_counts": {k: len(v) for k, v in split_patients.items()},
        "leakage_detected": has_leakage,
        "leakage_details": leakages,
        "missing_files_sample": missing_files,
    }


def create_patient_split(
    manifest_rows: List[Dict[str, Any]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Assigns 'split' column (train, val, test) using GroupShuffleSplit on patient_id.
    Strictly prevents patient leakage across subsets.
    """
    patient_ids = np.array([r["patient_id"] for r in manifest_rows])
    indices = np.arange(len(manifest_rows))

    # First split off test set by patient
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=patient_ids))

    # Next split train and val by patient
    relative_val_size = val_size / (1.0 - test_size)
    train_val_patients = patient_ids[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_sub_idx, val_sub_idx = next(gss_val.split(train_val_idx, groups=train_val_patients))

    final_train_idx = train_val_idx[train_sub_idx]
    final_val_idx = train_val_idx[val_sub_idx]

    updated_rows = [dict(r) for r in manifest_rows]
    for idx in final_train_idx:
        updated_rows[idx]["split"] = "train"
    for idx in final_val_idx:
        updated_rows[idx]["split"] = "val"
    for idx in test_idx:
        updated_rows[idx]["split"] = "test"

    return updated_rows
