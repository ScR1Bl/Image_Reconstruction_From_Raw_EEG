# Baseline 200-way — oficjalny benchmark THINGS-EEG2 (Step 0.5.1 + 0.5.2, 2026-09-01)

Surowe dane w tym katalogu: `report.json` + `config.json` (referencje linear/tsconv,
trenowane od zera w benchmarku); `official_report_semantic_encoder.json` +
`eval_config_semantic_encoder.json` (WGRANE WAGI enkodera semantycznego).

## ⭐ WYNIK GŁÓWNY — SubjectAdaptiveSemanticEncoder (wgrane wagi, Step 0.5.2)

To jest liczba odniesienia dla refaktoru Step 1. Ensemble 10 podmiotów, 80 repetycji:

| model | top-1 | top-5 | top-10 | MRR | median rank | chance | null perm. top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **semantic_encoder (wgrane wagi)** | **0.540** | **0.825** | 0.905 | 0.676 | **1.0** | 0.005 | 0.0047 ± 0.0040 |
| linear (referencja z benchmarku) | 0.415 | 0.750 | 0.865 | 0.570 | 2.0 | 0.005 | 0.0047 ± 0.0054 |

Delta enkoder − linear: **top-1 +0.125, top-5 +0.075** (wbudowane porównanie trenera:
"improved"). Przy 1 repetycji: top-1 0.206 / top-5 0.461. Per-subject (bez ensemble,
80 rep.): 0.240–0.390 (S10=0.390, S8=0.365).

Komenda (bez modyfikacji kodu — tryb eval trenera):

```powershell
.venv\Scripts\python.exe -m eeg_mae.training.train_semantic_encoder `
  --eval-only --resume artifacts/eeg_encoder/baseline/checkpoints/semantic_encoder.pt `
  --baseline-report runs/eeg_semantic_encoder/benchmark/report.json `
  --output runs/eeg_semantic_encoder/eval_baseline_weights
```

Uwaga metodologiczna: `official_200way_benchmark.py` NIE wspiera oceny zewnętrznego
checkpointu (`--models` ogranicza do linear/linear_full/tsconv — linie 32-37);
powyższa ścieżka to `train_semantic_encoder --eval-only`, który używa tej samej
agregacji co benchmark (komentarz train_semantic_encoder.py:209) i tych samych
banków/cache/seedów (seed 20260713, repeat_draws 8, 100 permutacji nulla).

## Komenda (dokładna)

```powershell
.venv\Scripts\python.exe -m eeg_mae.evaluation.official_200way_benchmark
```

(wszystkie flagi domyślne: --models linear tsconv, --subjects 0..9, --epochs 120,
--patience 18, --batch-size 512, --lr 3e-4, --seed 20260713, --device cuda,
banki data/derived/*_dinov2s_192.pt, indeks z eeg-build-index)

- Kod: **commit b9360cc** (pierwszy commit repo, wykonany przez człowieka po przebiegach;
  kod `eeg_mae/` i pyproject w chwili przebiegów identyczne ze stanem tego commita —
  między przebiegami a commitem zmieniały się tylko docs/vault/konfiguracja).
- Środowisko: Python 3.13.7, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
  RTX 5070 Ti (sm_120).
- Banki targetów zbudowane lokalnie wg docs/bank-recipes.md (recepta nr 1 i 2,
  --projected-dim 192); dane: THINGS-EEG2 preprocessed, 10 podmiotów.

## Referencje wbudowane benchmarku (Step 0.5.1) — ensemble 10 podmiotów, 80 repetycji

| metoda | top-1 | top-5 | top-10 | MRR | median rank | chance top-1 | null perm. top-1 (100 perm.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **linear** | **0.415** | **0.750** | 0.865 | 0.570 | 2.0 | 0.005 | 0.0047 ± 0.0054 |
| tsconv | 0.400 | 0.740 | 0.870 | 0.557 | 2.0 | 0.005 | 0.0049 ± 0.0055 |

Null permutacyjny top-5: linear 0.0248 ± 0.0116; tsconv 0.0265 ± 0.0122;
median rank nulla ≈ 100.7 (czyli losowy). Sygnał bije nulla o ~2 rzędy wielkości.

## Ensemble przy 1 repetycji (dolna kotwica protokołu)

linear: top-1 0.195 / top-5 0.4625; tsconv: top-1 0.159 / top-5 0.4025.

## Per-subject, top-1 @ 80 repetycji (pojedynczy podmiot, bez ensemble)

- linear: min 0.190 (S7) / max 0.385 (S8) / średnia 0.255
- tsconv: min 0.150 / max 0.370 / średnia 0.218

(numeracja S = subject_id+1; pełne krzywe repeat_counts 1..80 w report.json)

## Split-half reliability (kontrola jakości sygnału, per podmiot)

raw_poststim Spearman-Brown: 0.78-0.94 (min S9=0.780, max S10=0.939) — sygnał testowy
rzetelny u wszystkich podmiotów.

## Zastrzeżenia interpretacyjne

- Pole `verdict` w report.json ("use_as_semantic_initialization") jest ZAHARDKODOWANE
  w kodzie benchmarku (flaga C18 z AUDIT.md) — ignorować, liczą się surowe liczby wyżej.
- Benchmark pisze do `runs/eeg_semantic_encoder/benchmark/` (flaga C21) — stamtąd
  skopiowano report.json/config.json; checkpointy baseline'ów (linear/tsconv_subject_XX.pt
  + history) zostały w runs/.
- Wejście = ensemble 10 podmiotów, jak w treningu; krzywa N=1,3,5,10 z decyzji Q4
  przychodzi dopiero ze Step 1 (obecny kod jej nie umie).
- Bank _192 odtworzony empirycznie (--projected-dim 192); brak banku produkcyjnego
  do porównania statystyk — szczegóły docs/bank-recipes.md.
