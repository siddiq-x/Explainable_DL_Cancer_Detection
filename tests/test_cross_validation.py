"""CV regressions: isolated patients, best-weight scoring and training-only weights."""

import logging

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.data.splits import create_stratified_group_kfold
from src.models.trainer import Trainer, cross_validate


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 1)

    def forward(self, images):
        return self.head(images.mean(dim=(2, 3)))

    def predict_proba(self, images):
        return self(images).sigmoid()


def test_cross_validation_uses_inner_validation_and_restores_best(tmp_path, monkeypatch):
    import src.evaluation.evaluator as evaluator

    rows = []
    for patient in range(60):
        label = "MALIGNANT" if patient % 3 == 0 else "BENIGN"
        path = f"p{patient}.png"
        Image.new("RGB", (16, 16), (200 if label == "MALIGNANT" else 50,) * 3).save(tmp_path / path)
        for view in range(2):
            rows.append({"patient_id": f"P_{patient:03d}", "image_path": path,
                         "roi_mask_path": "", "label": label, "view": view})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = {
        "seed": 42, "experiment_name": "tiny_cv",
        "data": {"image_size": 16, "batch_size": 16, "num_workers": 0, "val_split_ratio": 0.2},
        "training": {"epochs": 1, "checkpoint_dir": str(tmp_path / "checkpoints"),
                     "learning_rate": 0.01, "loss": {"type": "bce", "class_weighting": True}},
    }
    original_train = Trainer.train
    original_collect = evaluator.collect_predictions
    selected_states = []
    models = []
    train_patients_by_fold = []

    def train_then_corrupt_final_weights(trainer):
        train_frame = trainer.train_loader.dataset.dataframe
        labels = train_frame["label"]
        expected_weight = (labels == "BENIGN").sum() / (labels == "MALIGNANT").sum()
        assert trainer.criterion.pos_weight.item() == pytest.approx(expected_weight)
        history = original_train(trainer)
        selected_states.append({key: value.clone() for key, value in trainer.model.state_dict().items()})
        train_patients_by_fold.append(set(train_frame["patient_id"]))
        # Final weights deliberately differ from the selected checkpoint.
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.add_(100)
        return history

    def check_held_out_inference(model, loader, **kwargs):
        idx = len(models)
        models.append(model)
        assert not train_patients_by_fold[idx] & set(loader.dataset.dataframe["patient_id"])
        for name, value in model.state_dict().items():
            assert torch.equal(value, selected_states[idx][name])
        return original_collect(model, loader, **kwargs)

    monkeypatch.setattr(Trainer, "train", train_then_corrupt_final_weights)
    monkeypatch.setattr(evaluator, "collect_predictions", check_held_out_inference)
    result = cross_validate(config, TinyClassifier, str(manifest), str(tmp_path), k=3,
                            device="cpu", logger=logging.getLogger("test_cv"), output_dir=str(tmp_path / "reports"))
    assert len({id(model) for model in models}) == 3
    assert "split_patient_ids" not in config
    held_out = []
    for fold in result["folds"]:
        patients = fold["patients"]
        train, val, test = (set(patients[key]) for key in ("train", "val", "test"))
        assert not (train & val or train & test or val & test)
        assert len(train | val | test) == 60
        assert fold["n_test"] == 40 and fold["best_epoch"] == 1
        held_out.extend(test)
        checkpoint = torch.load(fold["best_checkpoint"], weights_only=False)
        assert checkpoint["config"]["split_patient_ids"] == patients
    assert len(held_out) == len(set(held_out)) == 60
    expected = np.mean([fold["metrics"]["roc_auc"] for fold in result["folds"]])
    assert result["summary"]["roc_auc_mean"] == pytest.approx(expected)


def test_kfold_stratifies_patients_and_is_repeatable(tmp_path):
    frame = pd.DataFrame([
        {"patient_id": f"P_{i:03d}", "label": "MALIGNANT" if i % 2 else "BENIGN"}
        for i in range(40) for _ in range(2)
    ])
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    folds = create_stratified_group_kfold(str(manifest), k=5)
    repeated = create_stratified_group_kfold(str(manifest), k=5)
    for (train, val), (train_again, val_again) in zip(folds, repeated):
        pd.testing.assert_frame_equal(train, train_again)
        pd.testing.assert_frame_equal(val, val_again)
        assert val.drop_duplicates("patient_id")["label"].value_counts().to_dict() == {"BENIGN": 4, "MALIGNANT": 4}


def test_last_checkpoint_has_updated_best_score_and_full_history(tmp_path, monkeypatch):
    model = TinyClassifier()
    loader = [(torch.zeros(2, 3, 4, 4), torch.tensor([0., 1.]))]
    config = {"experiment_name": "selection", "training": {
        "epochs": 3, "early_stopping_patience": 1, "checkpoint_dir": str(tmp_path),
    }}
    trainer = Trainer(model, loader, loader, config, device="cpu", logger=logging.getLogger("test_selection"))
    validation_losses = iter([0.4, 0.6])

    def controlled_epoch(loader, train):
        if train:
            with torch.no_grad():
                model.head.bias.add_(1)
        return (0.3 if train else next(validation_losses), 0.5, np.array([0, 1]), np.array([0.2, 0.8]))

    monkeypatch.setattr(trainer, "_run_epoch", controlled_epoch)
    trainer.train()
    best = torch.load(tmp_path / "selection/best.pt", weights_only=False)
    last = torch.load(tmp_path / "selection/last.pt", weights_only=False)
    assert best["best_score"] == last["best_score"] == 0.4
    assert best["epoch"] == 0 and last["epoch"] == 1
    assert len(last["history"]["val_loss"]) == 2
    assert last["epochs_no_improve"] == 1
    assert not torch.equal(best["model_state_dict"]["head.bias"], last["model_state_dict"]["head.bias"])
