# Recepty budowy banków targetów

Dokładne komendy odtworzenia każdego artefaktu pochodnego. Zapisane w Step 0.5 (2026-09-01),
bo flagi produkcyjne nie były nigdzie w repo (AUDIT.md, flaga C2). Kolejne banki
(CLIP-emb, latenty VAE dla DiT) dopisywać tu na tym samym wzorcu.

Konwencja wpisu: komenda + wymagania + status recepty
(**ZWERYFIKOWANA** = flagi wprost z defaultów kodu, z odnośnikiem `plik:linia`;
**EMPIRYCZNA** = flagi zrekonstruowane, wymagają potwierdzenia porównaniem z artefaktem
produkcyjnym).

Wymagania wspólne: dataset w `data/things_eeg2_osf/` (layout: docs/data.md), venv
z `pip install -e ".[visual-teachers]"`, uruchamianie z korzenia repo (wszystkie defaulty
ścieżek są względne — AUDIT.md flaga C17). Modele HF pobierane z sieci (pinned rewizje
w kodzie builderów).

## 0. Indeks + cache EEG

```powershell
.venv\Scripts\eeg-build-index.exe    # data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv
.venv\Scripts\eeg-build-cache.exe    # data/derived/eeg_float32_cache/sub-XX__preprocessed_eeg_training.npy
```
ZWERYFIKOWANA (pyproject.toml:27-28; defaulty build_dataset_index.py:63, build_eeg_cache.py:19).
Uwaga: cache tworzy też lazily `cached_array` przy pierwszym treningu/benchmarku
(official_200way_benchmark.py:75-91) — jawne `eeg-build-cache` jest opcjonalne.

## 1. visual_targets_dinov2s_192.pt (bank DINO treningowy)

```powershell
.venv\Scripts\python.exe -m eeg_mae.data.build_visual_targets `
  --projected-dim 192 `
  --output data/derived/visual_targets_dinov2s_192.pt
```
**EMPIRYCZNA co do flag, KSZTAŁTY ZWERYFIKOWANE (Step 0.5.1, 2026-09-01)** —
defaulty buildera to projected-dim 64 i output bez sufiksu `_192`
(build_visual_targets.py:22,25), a wszyscy konsumenci oczekują `_192`
(np. train_semantic_encoder.py:29, official_200way_benchmark.py:24). Bank zbudowany
tą komendą przechodzi weryfikację: `dino_grid (16540, 8, 8, 192)` fp16,
`dino_global (16540, 384)` fp16, `color (16540, 63)` fp32, `projection (384, 192)`,
`metadata: {grid_size: 8, projected_dim: 192, seed: 2026, complete: True}`;
build_official_test_targets i benchmark konsumują go bez błędów. Porównanie statystyk
z bankiem PRODUKCYJNYM nie było możliwe — bundle wag nie zawierają banków, a banku
produkcyjnego nie ma lokalnie (pozostaje niepewność co do niedomyślnych flag innych
niż projected-dim, choć żadna inna nie zmienia kształtów).
UWAGA ŚRODOWISKOWA: transformers>=5.0 wymaga torchvision jako backendu
AutoImageProcessor (wymóg runtime, niewidoczny w grafie importów) — torchvision musi
być w `[visual-teachers]` (przywrócony w Step 0.5.1).

## 2. official_test_targets_dinov2s_192.pt (bank DINO oficjalnego testu 200)

```powershell
.venv\Scripts\python.exe -m eeg_mae.data.build_official_test_targets `
  --training-bank data/derived/visual_targets_dinov2s_192.pt
```
ZWERYFIKOWANA co do mechanizmu — grid_size/projected_dim czyta z metadanych banku
treningowego (build_official_test_targets.py:52-53), default output już z sufiksem `_192`
(build_official_test_targets.py:20). Zależy od poprawności recepty nr 1.

## 3. rich_visual_targets_32.pt (teachery geometrii 32×32)

```powershell
.venv\Scripts\python.exe -m eeg_mae.data.build_visual_geometry_bank
```
ZWERYFIKOWANA (defaulty build_visual_geometry_bank.py; teachery pinned:
Depth-Anything-V2-Small rev 5426e4f :29-30, BiRefNet_lite rev 7838f1c :31-32).

## 4. edge_targets_64_v2.pt (bazowe targety krawędzi 64×64)

```powershell
.venv\Scripts\python.exe -m eeg_mae.data.build_edge_base_targets
```
ZWERYFIKOWANA (defaulty build_edge_base_targets.py:30-36). Wymaga cv2+scipy
(importy top-level :10,:14 — moduł nie zaimportuje się bez `[visual-teachers]`).

## 5. pyramid_edge_targets_v1.pt (piramida krawędzi 16/32/64)

```powershell
# wymaga zewnętrznego checkoutu PiDiNet + checkpointu (spoza pip; AUDIT flaga C24):
#   git clone https://github.com/hellozhuo/pidinet data/external/pidinet
#   + data/external/pidinet/trained_models/table5_pidinet.pth
.venv\Scripts\python.exe -m eeg_mae.data.build_pyramid_edge_targets
```
ZWERYFIKOWANA co do flag (defaulty build_pyramid_edge_targets.py:47-55: wejścia
edge_targets_64_v2.pt + rich_visual_targets_32.pt, pidinet-root data/external/pidinet).
Zależy od recept 3 i 4.

## 6. rich_dino_color_v2/ (memmap cech DINO 4 warstw dla color oracle)

```powershell
.venv\Scripts\python.exe -m eeg_mae.data.build_rich_dino_color_bank
```
ZWERYFIKOWANA (defaulty build_rich_dino_color_bank.py:28-31; warstwy [3,6,9,12]
dinov2-small, patches fp16 (N,4,16,16,384), rgb uint8 64×64).

## 7. full_color_latents_antiring/ (latenty zamrożonego color oracle)

```powershell
# wymaga WYTRENOWANEGO color_oracle.pt (trener: eeg_mae/training/train_color_oracle.py)
.venv\Scripts\python.exe -m eeg_mae.data.build_full_color_latent_bank
```
ZWERYFIKOWANA co do flag (defaulty build_full_color_latent_bank.py:21-22: teacher
runs/eeg_color_encoder/components/color_oracle.pt, wejście rich_dino_color_v2).
Zależy od recepty 6 + wag oracle (z treningu albo z bundle'a release models-v1).

## Kolejność pełnego odtworzenia od zera

0 → 1 → 2 → 3 → 4 → 5 → 6 → [trening color_oracle] → 7.
Gałąź krawędzi potrzebuje 0,1,3,4,5; gałąź koloru 0,1,6,7; semantyka 0,1,2.
