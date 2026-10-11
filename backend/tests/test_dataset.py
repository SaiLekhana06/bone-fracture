import pytest
from backend.ml.dataset import create_patient_split

def test_patient_level_splitting_no_leakage():
    # Simulate a manifest of 20 samples from 8 distinct patients
    manifest = [
        {"image_path": f"img_{i}.png", "patient_id": f"P{i % 8:03d}", "label": i % 2}
        for i in range(40)
    ]

    split_manifest = create_patient_split(manifest, val_size=0.25, test_size=0.25, random_state=42)

    train_patients = set(r["patient_id"] for r in split_manifest if r["split"] == "train")
    val_patients = set(r["patient_id"] for r in split_manifest if r["split"] == "val")
    test_patients = set(r["patient_id"] for r in split_manifest if r["split"] == "test")

    # Strict isolation verification: intersection must be empty
    assert len(train_patients.intersection(val_patients)) == 0, "Patient overlap between train and val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient overlap between train and test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient overlap between val and test!"
