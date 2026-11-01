import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from eeg_mae.lightning.ensemble_trunk import EnsembleTrunkLightning

CONFIG = {
    "output_dim": 8,
    "subjects": 1,
    "channels": 2,
    "samples": 16,
    "input_samples": 20,
    "post_stimulus_start": 4,
    "temporal_features": 2,
    "dropout": 0.0,
}


def test_single_subject_fit_resume_self_contained(tmp_path):
    model = EnsembleTrunkLightning(trunk_config=CONFIG, n_max=1, n1_prob=0.0, base_lr=0.01)
    before = model.trunk.linear_weight.detach().clone()
    data = DataLoader(TensorDataset(torch.randn(4, 1, 2, 20), torch.randn(4, 8)), batch_size=2)
    trainer = L.Trainer(
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(model, data)
    assert not torch.equal(before, model.trunk.linear_weight)
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    # Historical hyperparameters may name a now absent initialization artifact.
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["hyper_parameters"]["trunk_checkpoint"] = str(tmp_path / "absent.ckpt")
    torch.save(checkpoint, path)
    restored = EnsembleTrunkLightning.load_from_checkpoint(path)
    assert torch.equal(restored.trunk.linear_weight, model.trunk.linear_weight)
    resumed = L.Trainer(
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    resumed.fit(restored, data, ckpt_path=path)
    assert resumed.global_step == 2
    assert resumed.optimizers[0].state_dict()["state"]


def test_subject_order_is_preserved():
    model = EnsembleTrunkLightning(trunk_config={**CONFIG, "subjects": 3}, n_max=3).eval()
    eeg = torch.randn(2, 2, 2, 20)
    ids = torch.tensor([2, 0])
    actual = model.embed_subjects(eeg, ids)
    for col, subject in enumerate(ids):
        assert torch.allclose(
            actual[:, col], model.trunk(eeg[:, col], subject.expand(2)), atol=1e-6
        )


def test_ensemble_dataset_repetition_pairing():
    import numpy as np

    from eeg_mae.lightning.data import EnsembleEEGDataset

    arrays = [
        np.arange(3 * 2 * 2 * 4, dtype=np.float32).reshape(3, 2, 2, 4) + 100 * s for s in range(2)
    ]
    targets = torch.tensor([[9.0], [3.0]])
    data = EnsembleEEGDataset(arrays, np.array([2, 0]), targets)
    eeg, target = data[0]
    assert torch.equal(target, targets[0])
    for subject in range(2):
        assert torch.equal(eeg[subject], torch.from_numpy(arrays[subject][2].mean(0)))


def test_preview_once_per_optimizer_step(tmp_path):
    from types import SimpleNamespace

    from eeg_mae.lightning.train_dit_adapter import SampleImages

    callback = SampleImages("", "", "", tmp_path, every=2)
    callback._val_clip = torch.zeros(1, 768)
    calls = []

    class Preview:
        device = torch.device("cpu")

        def eval(self):
            pass

        def train(self):
            pass

        def sample_images(self, *args, **kwargs):
            calls.append(1)
            return torch.zeros(1)

        def decode_and_grid(self, latents):
            from PIL import Image

            return Image.new("RGB", (1, 1))

    for _ in range(4):
        callback.on_train_batch_end(SimpleNamespace(global_step=2), Preview(), None, None, 0)
    assert len(calls) == 1
