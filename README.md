# GSN_EEG — odtwarzalny baseline SSSHANKS

Trzy pozostawione ścieżki dla THINGS-EEG2:

```text
EEG → enkoder semantyczny → CLIP 768 → adapter → PixArt → obraz
EEG → predyktor DINO → baza koloru + specjalista EEG → RGB 64×64
EEG → predyktor DINO → baza krawędzi + specjalista EEG → piramida 16/32/64
```

Baza kodu: `origin/s_dev`, commit `89142c2` autorstwa SSSHANKS, z ograniczonymi
poprawkami uruchamiania, checkpointów i oceny. Kolor i krawędzie pozostają osobnymi
wynikami; oryginalny adapter PixArt przyjmuje wyłącznie CLIP, nie te mapy.
Nie ma tutaj późniejszych eksperymentów hierarchicznych, hybryd, klasyfikatora
kategorii, referencji treningowych ani protokołu jednej osoby 6/2/2.

## Uruchomienie

Python 3.10+; dla generowania i pełnego treningu zalecany jest CUDA PyTorch
odpowiedni dla zainstalowanego sterownika. Z katalogu projektu:

```powershell
python -m pip install -e ".[dev,visual-teachers,dit]"
$pixart = "data/pretrained/models--PixArt-alpha--PixArt-XL-2-512x512/snapshots/50f702106901db6d0f8b67eb88e814c56ded2692"
```

Na tej maszynie istnieje `.venv`; można używać `.venv/Scripts/python.exe`
zamiast `python`. Pełna kolejność przygotowania, uczenia i oceny:
**[docs/reproduce.md](docs/reproduce.md)**. Format danych: [docs/data.md](docs/data.md).

Generowanie z istniejących wag, dla EEG jednej osoby (sub-01):

```powershell
python -m eeg_mae.evaluation.reconstruct_eeg --eeg input.npy --subject-ids 0 --repo $pixart --device cuda --output runs/reconstruction
```

Wejście ma kształt `[N,S,R,17,100]`: próbki obrazów, osoby, powtórzenia,
kanały, czas. Dla jednej osoby `S=1`; dla dziesięciu `S=10` i
`--subject-ids 0 1 2 3 4 5 6 7 8 9`. Identyfikatory odpowiadają kolejności
osób w pliku. Sygnał musi mieć preprocessing THINGS-EEG2. Enkoder wykorzystuje
80 próbek po bodźcu; powtórzenia są uśredniane. `--no-render` zapisuje same
warunki CLIP, bez ładowania PixArt.

Ten tor nie wyszukuje podobnego zdjęcia treningowego ani nie przewiduje promptu
kategorii. Generator dostaje rzeczywistą predykcję cech z EEG. Jakość wizualna
nie jest gwarantowana przez samo poprawne uruchomienie.

## Kolor i krawędzie

Oryginalne bundle zawierają komplet zależności zamrożonych modeli. Nie usuwać
ich katalogów `components/`.

```python
import torch
from eeg_mae import load_color_encoder, load_edge_encoder

color, _ = load_color_encoder("runs/eeg_color_encoder", "cpu")
edges, _ = load_edge_encoder("runs/eeg_edge_encoder", "cpu")
# eeg: [N,10,17,100], średnia powtórzeń, kolejność sub-01…sub-10
with torch.inference_mode():
    rgb = color(eeg).image.rgb
    crisp_edges = edges(eeg).pyramid.crisp_logits64.sigmoid()
```

Te historyczne modele oczekują pełnej kanonicznej kolejności dziesięciu osób.
Obsługa dowolnego podzbioru osób w nowym CLI dotyczy ścieżki semantycznej.

## Co znajduje się w projekcie

| Katalog | Zawartość |
|---|---|
| `eeg_mae/data` | Indeks, cache EEG, banki nauczycieli obrazowych |
| `eeg_mae/models` | Enkodery, dekodery i straty potrzebne trzem ścieżkom |
| `eeg_mae/training` | Oryginalne trenery DINO, koloru i krawędzi oraz ich komponentów |
| `eeg_mae/lightning` | Trening EEG→CLIP i adaptera PixArt |
| `eeg_mae/evaluation` | Benchmark 200-way, generowanie z EEG i ocena wygenerowanych obrazów |
| `data` | Dane źródłowe, potrzebne banki i cache modeli; poza Git |
| `runs` | Wybrane wagi; własne wyniki generowania; poza Git |
| `artifacts` | Nowe treningi Lightning; poza Git |

Historyczne konfiguracje i inicjalizacje zachowano obok wybranych wag.
Ich pochodzenie i sumy kontrolne: [models/manifest.json](models/manifest.json).
Zakres porządków i odzyskanie starego kodu: [docs/cleanup.md](docs/cleanup.md).

## Kontrola zmian

```powershell
python -m compileall -q eeg_mae
ruff format --check eeg_mae
```

Dane, checkpointy i cache pozostają poza repozytorium Git.
