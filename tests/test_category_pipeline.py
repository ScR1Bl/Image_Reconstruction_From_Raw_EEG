import numpy as np
import pytest
import torch

pytest.importorskip("lightning")

from eeg_mae.data.build_category_prior import retrieve, training_labels
from eeg_mae.data.build_category_text import category_prompt
from eeg_mae.data.build_text_comparison import text_for_predictions
from eeg_mae.evaluation.reconstruct_eeg import validate_eeg
from eeg_mae.lightning.category_classifier import select_reference


def test_reference_bank_rejects_held_out_images():
    rows = [{"split": "train", "zip_member": "images/001_cat/a.jpg"}]
    assert training_labels(rows)[0] == ["001_cat"]
    with pytest.raises(ValueError, match="only training"):
        training_labels(rows + [{**rows[0], "split": "val"}])


def test_reference_selected_within_predicted_class_without_ground_truth():
    bank = {
        "references": torch.eye(3),
        "labels": torch.tensor([0, 1, 1]),
        "rows": [{"split": "train"}] * 3,
    }
    # Query is closest to class 0, but the classifier predicts class 1.
    selected = select_reference(torch.tensor([[0.0, 2.0]]), torch.tensor([[1.0, 0.1, 0.2]]), bank)
    assert selected.item() == 2
    bank["rows"][2] = {"split": "test"}
    with pytest.raises(ValueError, match="non-training"):
        select_reference(torch.tensor([[0.0, 2.0]]), torch.ones(1, 3), bank)


def test_prototypes_and_nearest_references_obey_query_order():
    gallery = torch.eye(3)
    scores, selected, nearest = retrieve(gallery[[2, 0]], gallery, torch.arange(3), 3)
    assert scores.argmax(1).tolist() == [2, 0]
    assert selected.tolist() == nearest.tolist() == [2, 0]


def test_eeg_input_requires_all_subjects_repetitions_and_finite_samples():
    eeg = np.zeros((10, 4, 17, 100), dtype=np.float32)
    assert validate_eeg(eeg).shape == (1, 10, 4, 17, 100)
    with pytest.raises(ValueError, match="expected"):
        validate_eeg(eeg[:1])
    eeg[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_eeg(eeg)


def test_text_selection_ignores_true_category_and_image_filename():
    bank = {
        "category_names": ["a", "b"],
        "embeds": torch.tensor([[[1.0]], [[2.0]]]),
        "mask": torch.ones(2, 1),
    }
    records = [{"predicted_category": "b", "true_category": "a", "filename": "a.jpg"}]
    assert text_for_predictions(records, bank)["embeds"].item() == 2
    records[0]["true_category"] = "unknown"
    assert text_for_predictions(records, bank)["embeds"].item() == 2
    assert category_prompt("00005_air_conditioner") == "a photograph of a air conditioner"


def test_concept_prompts_preserve_distinct_word_senses():
    definitions = {
        "bass1": {
            "Word": "bass",
            "Top-down Category (manual selection)": "animal",
            "All Bottom-up Categories": "",
            "Definition (from WordNet, Google, or Wikipedia)": "a fish",
        },
        "bass2": {
            "Word": "bass",
            "Top-down Category (manual selection)": "musical instrument",
            "All Bottom-up Categories": "",
            "Definition (from WordNet, Google, or Wikipedia)": "a low pitched instrument",
        },
    }
    assert "fish" in category_prompt("00001_bass1", definitions)
    assert "instrument" in category_prompt("00002_bass2", definitions)


def test_text_condition_is_not_reused_for_unconditional_denoising():
    import lightning as L

    from eeg_mae.lightning.dit_adapter import DiTAdapterLightning, IPTokenHolder

    class Recorder(torch.nn.Module):
        def forward(self, value, **kwargs):
            self.text = kwargs["encoder_hidden_states"].clone()
            return type("Output", (), {"sample": torch.cat([value, value], 1)})()

    model = DiTAdapterLightning.__new__(DiTAdapterLightning)
    L.LightningModule.__init__(model)
    model.register_buffer("null_text", torch.zeros(1, 2, 3))
    model.register_buffer("null_mask", torch.ones(1, 2))
    model.holder = IPTokenHolder()
    model.transformer = Recorder()
    latent = torch.zeros(1, 4, 2, 2)
    timestep = torch.ones(1)
    prompt = torch.ones(1, 2, 3)
    model.denoise_forward(latent, timestep, None, prompt, torch.ones(1, 2))
    assert model.transformer.text.sum() == 6
    model.denoise_forward(latent, timestep, None)
    assert model.transformer.text.sum() == 0
    with pytest.raises(ValueError, match="both prompt"):
        model.denoise_forward(latent, timestep, None, prompt)
