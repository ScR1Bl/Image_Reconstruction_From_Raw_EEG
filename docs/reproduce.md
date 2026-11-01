# Odtworzenie eksperymentu

Wszystkie polecenia wykonuj z katalogu głównego projektu. `python` oznacza
interpreter z zainstalowanym `pip install -e ".[dev,visual-teachers,dit]"`.
Na obecnej maszynie użyj `.venv/Scripts/python.exe`. Nie uruchamiaj pełnych
treningów jednocześnie na jednej karcie.

## 1. Dane i cache

Wymagany układ opisano w [data.md](data.md). Używamy oficjalnie przetworzonego EEG,
17 kanałów, 100 próbek, z podziałem obrazów 8/1/1. Nie dzielimy powtórzeń tego
samego obrazu między train/val/test.

```powershell
python -m eeg_mae.data.build_dataset_index
python -m eeg_mae.data.build_eeg_cache
$pixart = "data/pretrained/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692"
if (-not (Test-Path $pixart)) { $pixart = "PixArt-alpha/PixArt-XL-2-512x512" }
```

Indeks, cache, obrazy i oficjalnie przetworzone EEG są już obecne lokalnie.
Surowe archiwum `data/raw/18470912.zip` usunięto na wyraźne życzenie użytkownika.
Bieżące trenery czytają `preprocessed/`, nie surowe napięcia z tego archiwum.

## 2. Cele obrazowe dla EEG→CLIP i PixArt

```powershell
python -m eeg_mae.data.build_clip_targets
python -m eeg_mae.data.build_vae_latents --repo $pixart
python -m eeg_mae.data.build_pixart_null --repo $pixart
```

Wyniki: `data/derived/clip_targets_vitl14_laion2b.pt`,
`vae_latents_pixart512.pt`, `pixart_null.pt`. Ostatni plik już zachowano lokalnie;
builder odmawia nadpisania, więc pomiń ten krok lub podaj inne `--output`.

CLIP-L/14 zwraca surowe cechy 768d. VAE zapisuje średnią i odchylenie posteriora;
latent jest losowany podczas treningu adaptera. Stały `null_learned` pochodzi
z zamrożonego checkpointu PixArt. Do tej ścieżki nie potrzeba T5 ani promptów kategorii.
Modele CLIP/PixArt mają przypięte rewizje w kodzie. `--max-images` w builderach
służy tylko do sprawdzenia działania; niepełny bank nie nadaje się do treningu.

## 3. Uczenie EEG→CLIP

Trening od losowej inicjalizacji, bez wcześniejszych wag EEG:

```powershell
python -m eeg_mae.lightning.train_eeg_encoder --run-name clip-scratch --fusion mean --target-key clip_global --output-dim 768 --n1-prob 1.0 --epochs 400 --val-every 5 --ensemble-base-lr 0.00002 --ensemble-residual-lr 0.0002
```

Wariant odpowiadający inicjalizacji wybranego historycznego modelu:

```powershell
python -m eeg_mae.lightning.train_eeg_encoder --run-name clip-warm --fusion mean --target-key clip_global --output-dim 768 --n1-prob 1.0 --epochs 400 --val-every 5 --ensemble-base-lr 0.00002 --ensemble-residual-lr 0.0002 --warm-start-checkpoint runs/eeg_encoder/initialization.ckpt
```

`n1-prob=1` oznacza uczenie na jednej losowanej osobie w danym batchu.
To model z parametrami osób i wspólną częścią sieci; nie dziesięć niezależnych
treningów. Walidacja raportuje wyniki poszczególnych osób oraz ich połączenia.
Przy generowaniu można użyć jednej osoby lub kilku prawdziwych identyfikatorów.

Wyniki trafiają do `artifacts/eeg_encoder/<run-name>/checkpoints/`.
`best.ckpt` wybierany jest walidacyjnie. Do wznowienia dodaj
`--resume artifacts/eeg_encoder/clip-warm/checkpoints/last.ckpt` do tej samej
komendy i zachowaj tę samą konfigurację. `--epochs` oznacza łączny docelowy limit.
`--smoke` służy krótkiemu sprawdzeniu treningu.

## 4. Uczenie adaptera PixArt

Adapter uczy się z **prawdziwych cech CLIP zdjęć treningowych** i latentów tych
samych zdjęć. Ten etap nie korzysta z EEG. Bazowy PixArt pozostaje zamrożony.

```powershell
python -m eeg_mae.lightning.train_dit_adapter --run-name pixart-scratch --repo $pixart --epochs 40 --batch-size 8 --accumulate 4
```

Odtworzenie startu historycznej końcowej fazy `faza1-t8c`:

```powershell
python -m eeg_mae.lightning.train_dit_adapter --run-name pixart-final --repo $pixart --init-from runs/dit_adapter/initialization.ckpt --epochs 12 --patience 6 --batch-size 8 --accumulate 4
```

To inicjalizacja wag, a nie pełne wznowienie: historyczny skrypt ignorował stan
optymalizatora, nawet gdy był obecny w pliku startowym. Zachowany `initialization.ckpt`
ma taki stan, natomiast historyczny końcowy `best.ckpt` go nie ma.
Nowe checkpointy zawierają stan treningu; je wznawiamy przez
`--resume artifacts/dit_adapter/pixart-final/checkpoints/last.ckpt` zamiast
`--init-from`. Nie łącz tych flag. `best.ckpt` wybierany jest po stracie walidacyjnej.

## 5. Generowanie z EEG

Przygotuj NumPy `[N,S,R,17,100]`, zgodny z kanałami i preprocessingiem datasetu.
Przykład eksportu pierwszych 8 obrazów walidacyjnych, wszystkich 10 osób:

```powershell
@'
import csv
import numpy as np
from pathlib import Path

with open('data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv', newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['subject_id']=='0' and r['repeat_index']=='0' and r['split']=='val'][:8]
ids = [int(r['image_index']) for r in rows]
eeg = np.stack([np.load(f'data/derived/eeg_float32_cache/sub-{s:02d}__preprocessed_eeg_training.npy', mmap_mode='r')[ids] for s in range(1,11)], axis=1)
Path('runs/input').mkdir(parents=True, exist_ok=True)
np.save('runs/input/validation.npy', eeg, allow_pickle=False)
Path('runs/input/filenames.txt').write_text('\n'.join(r['image_file'] for r in rows), encoding='utf-8')
'@ | python -

python -m eeg_mae.evaluation.reconstruct_eeg --eeg runs/input/validation.npy --subject-ids 0 1 2 3 4 5 6 7 8 9 --repo $pixart --device cuda --output runs/reconstruction
```

Domyślnie ładowane są zachowane `runs/eeg_encoder/best.ckpt` i
`runs/dit_adapter/best.ckpt`. Aby użyć nowego treningu, ustaw `--encoder` i
`--adapter` na odpowiednie pliki `artifacts/.../checkpoints/best.ckpt`.
Katalog wyjściowy musi być nowy. Wyniki: PNG, surowe warunki CLIP i raport parametrów.

Dla samego sub-01 wyeksportuj `eeg[:, :1]` do osobnego pliku i podaj
`--subject-ids 0`. Nie powielaj EEG jednej osoby do dziesięciu kolumn.
`--no-render --device cpu` sprawdza tor EEG→warunek bez PixArt.

## 6. Ocena po generowaniu

Nazwy bodźców służą dopiero do oceny; nie są wejściem generatora. Znajdź
odpowiadające im wiersze w kompletnym banku CLIP:

```powershell
@'
import torch
from pathlib import Path
bank = torch.load('data/derived/clip_targets_vitl14_laion2b.pt', map_location='cpu', weights_only=False)
lookup = {name: i for i, name in enumerate(bank['filenames'])}
print(' '.join(str(lookup[name]) for name in Path('runs/input/filenames.txt').read_text(encoding='utf-8').splitlines()))
'@ | python -
```

Przekaż wypisane numery jako wartości `--rows`:

```powershell
python -m eeg_mae.evaluation.evaluate_reconstruction --generated runs/reconstruction --targets data/derived/clip_targets_vitl14_laion2b.pt --rows 8 18 28 38 48 58 68 78 --device cuda
```

Liczby powyżej są przykładowe: obowiązuje kolejność nazw z faktycznie zbudowanego
banku. Raport zawiera cosinus wygenerowany-obraz/docelowy-CLIP i retrieval wśród
podanych kandydatów. Osiem obrazów to podgląd, nie miarodajny wynik całego zbioru.
Wiersze i ich przynależność do holdoutu wybiera wywołujący; sam scorer nie dowodzi
niezależności zestawu. Pełne wyniki wymagają uprzednio ustalonego zbioru i liczby kandydatów.

Oddzielny oficjalny benchmark 200-way enkodera:

```powershell
python -m eeg_mae.data.build_clip_targets --official-test --images-zip data/things_eeg2_osf/image_set/test_images.zip --output data/derived/official_clip_targets_vitl14_laion2b.pt
python -m eeg_mae.evaluation.eval_200way --checkpoint runs/eeg_encoder/best.ckpt --model ensemble-lightning --ensemble-n 1 10 --test-bank data/derived/official_clip_targets_vitl14_laion2b.pt --test-bank-key clip_global --output runs/eeg_encoder/official_200way.json
```

To wyszukiwanie obrazu na podstawie EEG, nie miara jakości obrazów z PixArt.

## 7. Zachowane gałęzie koloru i krawędzi

Są to oryginalne modele z zamrożonymi bazami DINO, nie późniejsze modele `direct`.
Lokalnie zachowano pełne `runs/eeg_color_encoder/components/` oraz
`runs/eeg_edge_encoder/components/`. Banki i zależności:

| Etap | Polecenie `python -m ...` | Wynik / zależność |
|---|---|---|
| DINO | `eeg_mae.data.build_visual_targets --projected-dim 192 --output data/derived/visual_targets_dinov2s_192.pt` | Globalny DINO i siatka |
| Geometria obrazów | `eeg_mae.data.build_visual_geometry_bank` | `rich_visual_targets_32.pt` |
| Krawędzie bazowe | `eeg_mae.data.build_edge_base_targets` | `edge_targets_64_v2.pt` |
| Piramida | `eeg_mae.data.build_pyramid_edge_targets` | Wymaga dwóch poprzednich banków i PiDiNet w `data/external/pidinet` |
| DINO dla koloru | `eeg_mae.data.build_rich_dino_color_bank` | `rich_dino_color_v2/` |
| Latenty koloru | `eeg_mae.data.build_full_color_latent_bank` | Wymaga banku powyżej i `components/color_oracle.pt` |

Już istniejących, kompletnych banków nie trzeba przeliczać. PiDiNet potrzebuje
`trained_models/table5_pidinet.pth`; jego kod i wagi są lokalnie zachowane.

Ponowny trening końcowych specjalistów z zachowanymi bazami:

```powershell
python -m eeg_mae.training.train_color_encoder --output runs/color-retrain
python -m eeg_mae.training.train_edge_encoder --output runs/edge-retrain
```

Trenery domyślnie używają komponentów z oryginalnych bundle. Aby nowe wyniki
otwierać publicznym loaderem, skopiuj do ich katalogów odpowiednie `components/`
z oryginalnego bundle, użyte przy treningu. Nigdy nie mieszaj komponentów różnych biegów.

Pozostałe trenery `train_color_decoder`, `train_color_oracle`, `train_edge_decoder`,
`train_edge_baseline` odtwarzają poszczególne komponenty; ich `--help` wskazuje
wejścia i wyjścia. Bank latentów koloru po zmianie oracle musi zostać zbudowany na nowo.

**Granica odtwarzalności:** gałąź SSSHANKS nie zawiera treningu historycznego
kolorowego `components/adapter.pt`. Dlatego zachowano ten artefakt jako konieczny
punkt startowy. Można odtworzyć wejście → trening specjalisty → wynik z zachowaną
bazą; nie należy twierdzić, że całe historyczne drzewo koloru da się wytrenować
od losowych wag wyłącznie z tego repo. Ścieżka EEG→CLIP→PixArt ma niezależny
wariant uczenia od losowej inicjalizacji.

## Różnice względem starego uruchomienia

- Naprawiono obsługę remisów retrieval i odtwarzanie samodzielnych checkpointów.
- Nowe wznowienie Lightning odtwarza optimizer; stare `--resume` adaptera było tylko inicjalizacją wag.
- Dostępne jest jawne generowanie z EEG; stary ewaluator adaptera podawał prawdziwy CLIP zdjęcia.
- Legacy loader semantyki działa jako jeden obsługiwany strumień batchy. To naprawa
  uruchamiania, nie gwarancja identycznej kolejności batchy dawnego treningu.
- Testy i smoke potwierdzają działanie kodu; pełne wieloepokowe treningi i odtworzenie
  historycznych liczb nie zostały wykonane podczas sprzątania.
