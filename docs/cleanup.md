# Porządki — 1 listopada 2026

Uporządkowana wersja znajduje się na `main`. Poprzedni `main`
(`677ee8d`) zachowano jako `old_main`. Bazą porządków jest `origin/s_dev`
na `89142c2`, z poprawkami odtwarzalności. Gałąź autora `s_dev` pozostaje bez zmian.
Gałąź `cleanup/ssshanks-reproduction` zachowuje stan przed usunięciem testów.

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

Po osobnym potwierdzeniu użytkownika usunięto również jedyną lokalną kopię
surowego EEG `data/raw/18470912.zip`: 117 109 620 289 bajtów (~117 GB).
Sprawdzono, że wszystkie 10 archiwów oficjalnie przetworzonego EEG pozostało.
Łącznie usunięto 820 plików, 152 099 693 595 bajtów (~152 GB).

## Zachowane

- Oficjalnie przetworzone EEG, obrazy, kod preprocessingu autorów i cache EEG.
- Banki DINO/koloru/krawędzi i PiDiNet potrzebne zachowanym ścieżkom.
- Oryginalne bundle koloru i krawędzi z ich katalogami `components`.
- Wybrany EEG→CLIP `clip2b-warm-lr2` jako `runs/eeg_encoder/best.ckpt`.
- Wybrany adapter `faza1-t8c` jako `runs/dit_adapter/best.ckpt`.
- Historyczne checkpointy inicjalizacyjne i konfiguracje tych dwóch przebiegów.
- Cache PixArt w `data/pretrained`, stały warunek tekstowy w `data/derived/pixart_null.pt`.

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

Przed usunięciem testów: 35 testów przeszło, a 62 pliki Python przeszły kontrolę formatowania.
Zachowane modele koloru i krawędzi przetworzyły prawdziwe EEG. Pełny test techniczny
EEG→CLIP→PixArt→VAE zapisał PNG 512×512 przy dwóch krokach generowania. Nie jest
to pomiar jakości ani ponowne przeprowadzenie pełnych treningów.

Następnie, na wyraźne życzenie użytkownika, z nowego `main` usunięto katalog
źródeł testów, workflow testowy i konfigurację pytest. Testy można odzyskać
z gałęzi `cleanup/ssshanks-reproduction`; nie są potrzebne do inferencji
ani treningu. Pozostawiono kontrolę składni i formatowania kodu.
Pozostały cache testów przeniesiono poza projekt do
`D:/GSN_EEG_archive_20261101/removed-tests-cache` i `removed-pytest-cache`,
ponieważ automatyczna kontrola zablokowała jego kasowanie. W katalogu projektu
nie ma już `tests/` ani `.pytest_cache/`.

Pusty katalog `runs/dit_ten_subjects/comparison_generation` był zablokowany przez
inny proces; nie zawiera wyników. Automatyczna kontrola narzędzi odrzuciła usunięcie
dwóch plików `.incomplete` w `data/pretrained/.../blobs/` (łącznie 3 617 587 200
bajtów). Nie są potrzebne do sprawdzonego renderowania, lecz pozostały na dysku.
