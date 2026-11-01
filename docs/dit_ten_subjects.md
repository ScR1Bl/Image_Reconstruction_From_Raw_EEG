# EEG dziesięciu osób → adapter kolegi → PixArt

Eksperyment z 13 września 2026 na branchu `research/eeg-dit-ten-subjects`.
Użytkownik jawnie potwierdził **fuzję EEG wszystkich 10 osób oglądających ten sam
obraz**. To osobny eksperyment względem wcześniejszego protokołu jednej osoby.

## Wagi i zgodność danych

Odnaleziono `D:/GSN_EEG/artifacts.tar` (7 805 395 968 bajtów), SHA-256
`8e0500fe87828578bc72ed3b4396e5fef9229a97d5841da7edadd7dada6db0f1`.
Po sprawdzeniu ścieżek, typów plików i wykluczeniu linków rozpakowano 306 wpisów
do `runs/colleague_artifacts_20260913/`. Archiwum i wagi pozostają poza Git.

Wykorzystano:

- `dit_adapter/faza1-t8c/checkpoints/best.ckpt` — ten sam checkpoint, który wskazuje
  raport `faza2/report.json` kolegi; 8 tokenów z CLIP 768d.
- `dit_adapter/t5_embeddings.pt`, wpis `null_learned` — stały warunek tekstowy,
  bez podawania nazw kategorii podczas generacji.
- `eeg_encoder/clip2b-warm-lr2/checkpoints/best.ckpt` — encoder EEG do CLIP-L/14.
- Lokalne, niezależne, wcześniej wytrenowane encodery 10 osób:
  `runs/eeg_edge_encoder_direct/best.pt` i `runs/eeg_color_encoder_direct/best.pt`.
  Nie duplikowano jednoosobowego EEG ani wag ostatniego eksperymentu sub-01.

Z repozytoriów autorów pobrano brakujący zamrożony model
[PixArt-XL-2-512x512](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512),
rewizja `50f702106901db6d0f8b67eb88e814c56ded2692`, tylko transformer, scheduler i VAE;
oraz wizualny scorer
[CLIP-L/14 LAION](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K),
rewizja `1627032197142fbe2a7cfec626f4ced3ae60d07a`.
Nie trzeba było pobierać ani uruchamiać dużego encodera T5.

Stan adaptera załadowano ściśle. Model bazowy nie wykorzystuje starego pola
`caption_projection.y_embedding`; zamiast niego wejście null pochodzi z dostarczonego
pliku T5. Oryginalne obrazy odtworzone scorerem CLIP mają zgodność cosinusową
>0,999997 z cechami obrazu z archiwum (8 sprawdzanych bodźców).

Ponownie obliczono lokalnie cechy EEG dla całego train i val. Minimalny cosinus
względem zapisu kolegi wynosi 0,99999952 dla train i 0,99999928 dla val, po wszystkich
osobach i obrazach. Potwierdza to zgodność kolejności obrazów, osób, checkpointu
i preprocessingu w granicach precyzji numerycznej. Hashy trzech użytych checkpointów
oraz wyników zgodności należy szukać w `conditions/manifest.json`.

## Protokół

Użyto starego podziału 8/1/1 obrazu w kategorii: 13232 train, 1654 val. W tym
eksperymencie nie oceniono wewnętrznego testu ani oficjalnych 200 nowych kategorii.
Każda osoba dostarcza średnią czterech prezentacji; okno EEG 0–790 ms (80 próbek).
Fuzja semantyki to średnia surowych wektorów 10 osób. Gałęzie struktury korzystają
z własnych, wyuczonych mechanizmów łączenia osób.

Walidacja służyła już wyborowi istniejących encoderów i służy wyborowi nowego
mapowania. Nie jest niezależnym testem końcowym. Brak potwierdzonej rozłączności
sekwencji prezentacji; późne EEG może zawierać odpowiedzi na następne bodźce.
Tych liczb nie porównujemy bezpośrednio z nowym podziałem 6/2/2 i jedną osobą.

## Jak podłączono encodery

Zamrożony adapter kolegi oczekuje wektora CLIP-L/14 o 768 współrzędnych, a nie
bezpośrednio siatek krawędzi i koloru. Dodano więc mały mostek uczony w Lightning:

```text
EEG 10 osób → encoder kolegi → średnia CLIP768 ───────┐
EEG 10 osób → Wasz encoder krawędzi → siatka 8×8×192 ├→ mostek → CLIP768
EEG 10 osób → Wasz encoder koloru → siatka 8×8×192 ──┘           ↓
                                                    zamrożony adapter → PixArt
```

Siatki 16×16 są zmniejszane do 8×8 przez average pooling; korzystamy z latentnych
warunków, nie z rozmytych RGB lub masek jako wejścia mostka. Cztery warianty:
S, S+E, S+C i S+E+C mają te same kształty parametrów (6,7 mln); wyłączone modality
są zerowane przed pierwszą warstwą. Wariant S ma jednak mniej aktywnych kolumn,
więc równa liczba parametrów nie oznacza identycznej efektywnej pojemności.

Mostek zaczyna od predykcji semantycznej z normą wyliczoną wyłącznie z train.
Uczy się resztowej korekty. Strata: MSE natywnego CLIP, skalowane przez
`768 / mean_training_norm²`, plus 0,1 symetrycznego InfoNCE. Checkpoint wybierany
według MSE walidacyjnego, maksymalnie 40 epok, patience 7. MRR monitorowany osobno.
Nie dobierano checkpointów na podstawie wyglądu wygenerowanych przykładów.

**To działający pełny tor EEG → obraz, ale z globalnym mostkiem do CLIP.**
Nie jest to nowy przestrzenny adapter DiT, ControlNet ani osobne K/V dla siatek.
Eksperyment nie rozstrzyga skuteczności bezpośredniego warunkowania przestrzennego.
Mostek uczy się na predykcjach encoderów dla ich własnego train, bez cross-fittingu;
to ograniczenie uczenia etapowego jest jawnie oznaczone w metadanych.

## Stwierdzony problem surowego wejścia EEG

Surowa semantyka 10 osób osiąga Top-1 14,21% i MRR 0,2645 w galerii 1654 obrazów,
ale bezpośrednia generacja daje głównie niezwiązane z bodźcem portrety i abstrakcje.
Kontrola z prawdziwym CLIP obrazu odtwarza znacznie bardziej odpowiednie obiekty.

W walidacji średnia norma predykcji EEG to 29,87, prawdziwego CLIP 20,44; cosinus
predykcja–obraz to 0,348. Sama zmiana normy nie zmienia cosinusa i nie naprawiła
generacji w panelu. Różnią się również średnie wektory zbioru. Dane wskazują na
niezgodność rozkładu wejścia adaptera; nie dowodzą, że sama norma jest jej przyczyną.

| Warunek EEG | Cosinus do prawdziwego CLIP | MSE CLIP | Top-1 w galerii 1654 |
|---|---:|---:|---:|
| Surowa semantyka | 0,3477 | 1,2527 | 14,21% |
| Semantyka, dopasowana norma | 0,3477 | 0,7100 | 14,21% |
| Mostek S | 0,6653 | 0,3629 | 12,39% |
| Mostek S+E | 0,6712 | 0,3565 | 10,22% |
| Mostek S+C | 0,6719 | 0,3557 | 12,64% |
| Mostek S+E+C | 0,6728 | 0,3548 | 10,04% |

MSE i cosinus się poprawiły kosztem rozróżniania obrazów. Nie oznacza to automatycznie
poprawy odtwarzania obiektu ani geometrii; wspólna składowa cech może podnosić
cosinus. Ocenę generacji zapisuje osobny `comparison_generation/scores.json`.

## Generacja i interpretacja panelu

Wybrano z góry pierwsze 8 obrazów starej walidacji — te same kategorie co na panelu
kolegi, bez wybierania najlepszych przykładów. To mały, niereprezentatywny pilot.
Każdy wariant tego samego obrazu ma identyczne początkowe ziarno `20260913 + i`,
20 kroków DPM-Solver, CFG 4,5, wyuczone wagi adaptera. Stały null zamiast etykiety.

`image_oracle` wykorzystuje cechy prawdziwego obrazu i stanowi wyłącznie kontrolę
generatora. Pozostałe warianty korzystają z EEG. `bridge_SEC_shuffled` zachowuje
semantykę, ale podaje E/C innych obrazów; jest kontrolą interwencji na warunkach.

Scorer porównuje wygenerowany obraz z **prawdziwymi** cechami wszystkich 1654
obrazów walidacyjnych. Nigdy nie traktuje przewidywanego ani pomieszanego wektora
jako prawdziwej referencji. CLIP jest równocześnie celem adaptera i miarą oceny,
więc ten pomiar nie jest niezależnym dowodem odzyskania geometrii.

## Pliki i odtworzenie

Wyniki lokalne w `runs/dit_ten_subjects/`:

- `baseline_generation/comparison.png`: oracle, surowe EEG, dopasowana norma.
- `encoder_preview.png`: maski i RGB obecnych encoderów 10 osób.
- `bridges/{S,S_E,S_C,S_E_C}/`: wagi, logi i metryki czterech treningów.
- `bridge_comparison.json`: metryki warunków na całej walidacji.
- `comparison_generation/comparison.png`: panel wszystkich wariantów.
- `comparison_generation/scores.json`: ocena wygenerowanych obrazów.
- `artifact_provenance.json`: hash archiwum i rewizje pobranych modeli.

```powershell
.venv/Scripts/python.exe -m eeg_mae.data.dit_ten_subjects
.venv/Scripts/python.exe -m eeg_mae.lightning.dit_condition_bridge --variant S_E_C --output runs/nowy_bridge
.venv/Scripts/python.exe -m eeg_mae.data.build_dit_comparison
.venv/Scripts/python.exe -m eeg_mae.evaluation.render_dit_conditions --conditions runs/dit_ten_subjects/comparison_conditions.pt --output runs/dit_ten_subjects/comparison_generation
.venv/Scripts/python.exe -m eeg_mae.evaluation.score_dit_generation
```

Katalogi ukończonych etapów nie powinny być nadpisywane. Renderer odmawia wznowienia
z innymi wagami, warunkami, ziarnem lub parametrami; obrazki zapisuje po każdym
przykładzie. Pierwszy panel baseline powstał przed dodaniem kontroli sygnatury,
dlatego jego ponowne wykonanie wymaga nowego katalogu wyjściowego.

Kod: `1bfbc32`. Walidacja kodu: **61 testów przeszło**; obejmuje identyczność mostka
z bazą przy inicjalizacji, wyłączenie modality, równą liczbę parametrów, uczciwe
remisy metryki oraz odrzucenie zmienionych ustawień wznowienia. Pierwsza generacja,
scorer i zgodność banków dodatkowo zostały sprawdzone na rzeczywistych plikach.


## Uko?czony pilot generacji (8 obraz?w)

| Warunek | Cosinus CLIP do bod?ca | Top-1 / 1654 | Top-5 / 1654 |
|---|---:|---:|---:|
| Prawdziwy obraz (oracle) | 0,6144 | 4/8 | 7/8 |
| Surowe EEG | 0,3104 | 0/8 | 0/8 |
| EEG z dopasowan? norm? | 0,3291 | 0/8 | 0/8 |
| Mostek S | 0,5285 | 1/8 | 1/8 |
| Mostek S+E | 0,5500 | 1/8 | 2/8 |
| Mostek S+C | 0,5222 | 1/8 | 2/8 |
| Mostek S+E+C | 0,5384 | 0/8 | 2/8 |
| S+E+C, przemieszane E/C | 0,4907 | 0/8 | 1/8 |

To nie jest dow?d przewagi pe?nego zestawu enkoder?w. Wida? lepsze przybli?enie
wygl?du po mostkach, lecz kategori? i cz??ci obiektu nadal cz?sto gubimy.
Kontrola oracle te? bywa b??dna, wi?c problem dotyczy tak?e samego adaptera.
Dalsze rozpoznawanie klas i sterowanie generatorem opisuje
[raport kategorii](category_generation.md).
