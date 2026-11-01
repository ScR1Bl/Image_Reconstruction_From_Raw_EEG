"""Train a subject-adaptive EEG encoder directly against raw CLIP image targets.

Mean/attention ensembles support scratch training or an explicit DINO warm start.
Legacy per-subject DINO training remains available for historical checkpoints."""

from __future__ import annotations

import argparse

import lightning as L

from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .data import SemanticDataModule
from .eeg_encoder import SemanticEncoderLightning

MODEL_NAME = "eeg_encoder"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG semantic trunk (Lightning)")
    add_run_arguments(parser)
    parser.add_argument("--training-bank", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Lightning .ckpt startowy (np. baseline-lightning trunk)",
    )
    parser.add_argument(
        "--baseline-dir", default=None, help="katalog linear_subject_XX.pt do inicjalizacji od zera"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=3e-5)
    parser.add_argument("--residual-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--base-freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    # Step 1b: fuzja podmiotow. "legacy" = trening per-subject jak w 1a;
    # "mean"/"attention" = ensemble z losowym N (EnsembleTrunkLightning).
    parser.add_argument("--fusion", choices=("legacy", "mean", "attention"), default="mean")
    parser.add_argument("--n1-prob", type=float, default=0.25)
    parser.add_argument(
        "--trunk-checkpoint",
        default=None,
        help="trunk startowy dla trybow ensemble (Lightning .ckpt)",
    )
    parser.add_argument("--fusion-lr", type=float, default=3e-4)
    parser.add_argument("--ensemble-base-lr", type=float, default=1e-5)
    parser.add_argument("--ensemble-residual-lr", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="zamroz trunk (N=1 bitowo = init); uczy sie tylko fuzja; monitor=val/mrr_n10",
    )
    # Step 2b: trunk pod inna przestrzen docelowa.
    parser.add_argument("--target-key", default="clip_global")
    parser.add_argument(
        "--output-dim",
        type=int,
        default=768,
        help="wymiar wyjscia trunku (wyklucza --trunk-checkpoint; np. 768 dla CLIP)",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        default=None,
        help="trunk-artefakt .ckpt: skopiuj czesci wspoldzielone o ksztaltach "
        "niezaleznych od output_dim (input_norm, channel_*, temporal_spatial, "
        "LayerNorm projekcji, residual_gate); reszta swieza",
    )
    parser.add_argument(
        "--resume", default=None, help="Full Lightning checkpoint, including optimizer"
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    if args.fusion != "legacy":
        from .common import sha256_of_file
        from .data import EnsembleDataModule
        from .ensemble_trunk import EnsembleTrunkLightning

        if args.output_dim:
            # Nowa przestrzen docelowa: trunk swiezy (opcjonalnie warm-start
            # czesci wspoldzielonych); pelny trunk-checkpoint nie pasuje ksztaltem.
            module = EnsembleTrunkLightning(
                trunk_config={"output_dim": args.output_dim},
                fusion=args.fusion,
                n1_prob=args.n1_prob,
                freeze_trunk=args.freeze_trunk,
                base_lr=args.ensemble_base_lr,
                residual_lr=args.ensemble_residual_lr,
                fusion_lr=args.fusion_lr,
                weight_decay=args.weight_decay,
            )
            if args.warm_start_checkpoint:
                from .common import load_trunk_state

                source = load_trunk_state(args.warm_start_checkpoint, None)
                shared = {
                    key: value
                    for key, value in source.items()
                    if key in module.trunk.state_dict()
                    and module.trunk.state_dict()[key].shape == value.shape
                }
                expected_prefixes = (
                    "input_norm_weight",
                    "input_norm_bias",
                    "channel_scale",
                    "channel_bias",
                    "temporal_spatial",
                    "temporal_project.0",
                    "residual_gate",
                )
                unexpected = [key for key in shared if not key.startswith(expected_prefixes)]
                if unexpected:
                    raise SystemExit(f"warm-start: nieoczekiwane pasujace klucze {unexpected}")
                missing, _ = module.trunk.load_state_dict(shared, strict=False)
                print(
                    f"warm-start: skopiowano {len(shared)} tensorow wspoldzielonych, "
                    f"{len(missing)} swiezych",
                    flush=True,
                )
        else:
            module = EnsembleTrunkLightning(
                trunk_checkpoint=args.trunk_checkpoint,
                trunk_sha256=sha256_of_file(args.trunk_checkpoint),
                fusion=args.fusion,
                n1_prob=args.n1_prob,
                freeze_trunk=args.freeze_trunk,
                base_lr=args.ensemble_base_lr,
                residual_lr=args.ensemble_residual_lr,
                fusion_lr=args.fusion_lr,
                weight_decay=args.weight_decay,
            )
        if args.resume:
            module = EnsembleTrunkLightning.load_from_checkpoint(args.resume, map_location="cpu")
        datamodule = EnsembleDataModule(
            training_bank=args.training_bank,
            target_key=args.target_key,
            index=args.index,
            archives=args.archives,
            cache=args.cache,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            smoke=args.smoke,
        )
        trainer = build_trainer(
            run_dir,
            args,
            max_epochs=args.epochs,
            gradient_clip_val=5.0,
            # przy zamrozonym trunku val/mrr_n1 jest stala - selekcja po n10
            monitor="val/mrr_n10" if args.freeze_trunk else "val/mrr_n1",
            patience=args.patience,
            mode="max",
            check_val_every=args.val_every,
        )
        trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)
        return
    if args.target_key != "dino_global":
        raise SystemExit("legacy mode requires --target-key dino_global and a DINO bank")
    if args.init_checkpoint:
        module = SemanticEncoderLightning.load_from_checkpoint(
            args.init_checkpoint, map_location="cpu"
        )
    else:
        module = SemanticEncoderLightning(
            base_lr=args.base_lr,
            residual_lr=args.residual_lr,
            weight_decay=args.weight_decay,
            base_freeze_epochs=args.base_freeze_epochs,
            max_epochs=args.epochs,
        )
        if args.baseline_dir:
            module.model.load_linear_baselines(args.baseline_dir)
    if args.resume:
        module = SemanticEncoderLightning.load_from_checkpoint(args.resume, map_location="cpu")
    datamodule = SemanticDataModule(
        training_bank=args.training_bank,
        index=args.index,
        archives=args.archives,
        cache=args.cache,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        smoke=args.smoke,
    )
    trainer = build_trainer(
        run_dir,
        args,
        max_epochs=args.epochs,
        gradient_clip_val=5.0,
        monitor="val/loss",
        patience=args.patience,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
