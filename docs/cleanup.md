# Porządki — 1 listopada 2026

Gałąź robocza: `cleanup/ssshanks-reproduction`. Bazą jest `origin/s_dev` na
`89142c2`, z poprawkami odtwarzalności. `main` i gałęzie autora nie były zmieniane.

## Usunięte

- Późniejsze encodery hierarchiczne, fusion prior, hybrydy koloru, protokół
  jednej osoby, mostki eksperymentalne i generowanie sterowane kategorią/referencją.
- Wyniki i wagi tych prób, warianty SigLIP/EVA i głowice porównawcze CLIP.
- Zdublowane opakowania treningu koloru/krawędzi w Lightning; zachowano ich
  oryginalne trenery z `training/` i wymagane komponenty.
- Historyczne raporty badawcze z aktywnego katalogu oraz testy usuniętych funkcji.
- Archiwum `artifacts.tar` po wybraniu i sprawdzeniu potrzebnych checkpointów.

Trwale usunięto 819 plików artefaktów, 34 990 073 306 bajtów (~35 GB).
Szczegółowy wykaz usuniętych plików pozostaje poza projektem w
`D:/GSN_EEG_archive_20261101/deleted-artifacts.csv`.

## Zachowane

- Źródłowy dataset, obrazy, preprocessing autorów i cache EEG.
- Banki DINO/koloru/krawędzi i PiDiNet potrzebne zachowanym ścieżkom.
- Oryginalne bundle koloru i krawędzi z ich katalogami `components`.
- Wybrany EEG→CLIP `clip2b-warm-lr2` jako `runs/eeg_encoder/best.ckpt`.
- Wybrany adapter `faza1-t8c` jako `runs/dit_adapter/best.ckpt`.
- Historyczne checkpointy inicjalizacyjne i konfiguracje tych dwóch przebiegów.
- Cache PixArt w `data/pretrained`, stały warunek tekstowy w `data/derived/pixart_null.pt`.
- Surowe EEG `18470912.zip` przeniesione do `data/raw/`; to dane źródłowe, nie zbędny wynik.

## Odzyskanie poprzedniego kodu

Kompletny stan kodu, również wcześniejsze niezacommitowane zmiany, znajduje się
na gałęzi `archive/pre-cleanup-20261101`, commit `4bf6e6c`. Dodatkowa kopia Git:
`D:/GSN_EEG_archive_20261101/history.bundle`. Jest tam także patch i kopia plików,
które przed sprzątaniem nie były śledzone.

Przykład odczytu starego pliku bez zmieniania katalogu roboczego:

```powershell
git show archive/pre-cleanup-20261101:eeg_mae/models/hierarchical_semantic.py
```

Kopia Git nie zawiera usuniętych wag i wyników. Ich trwałe usunięcie zostało
wybrane przez użytkownika. Nie obiecuje się ich odzyskania z archiwum kodu.

## Weryfikacja i pozostałości

Po zmianach: 35 testów przeszło; 62 pliki Python przechodzą kontrolę formatowania.
Zachowane modele koloru i krawędzi przetworzyły prawdziwe EEG. Pełny test techniczny
EEG→CLIP→PixArt→VAE zapisał PNG 512×512 przy dwóch krokach generowania. Nie jest
to pomiar jakości ani ponowne przeprowadzenie pełnych treningów.

Pusty katalog `runs/dit_ten_subjects/comparison_generation` był zablokowany przez
inny proces; nie zawiera wyników. Automatyczna kontrola narzędzi odrzuciła usunięcie
dwóch plików `.incomplete` w `data/pretrained/.../blobs/` (łącznie 3 617 587 200
bajtów). Nie są potrzebne do sprawdzonego renderowania, lecz pozostały na dysku.
