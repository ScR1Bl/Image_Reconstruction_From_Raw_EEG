# Przegląd projektu i zmian SSSHANKS

Data: 12 września 2026. Repozytorium: `ScR1Bl/Image_Reconstruction_From_Raw_EEG`.

## Wersje i zakres

SSSHANKS jest autorem commitów. Gałąź to `origin/s_dev`, ze szczytem `89142c2`. Po `git fetch origin` historia zawiera:

| Commit | Data autora | Opis |
|---|---|---|
| `b9360cc` | 2026-09-01 | pre-refactor baseline |
| `d36519c` | 2026-09-01 | post-refactor, behavior-preserving |
| `23d83f8` | 2026-09-02 | fusion |
| `89142c2` | 2026-09-08 | DiT adapter first trial |

Lokalny `main` i `origin/main` wskazują `677ee8d`. Wszystkie cztery commity SSSHANKS są jego potomkami, ale nie zostały scalone do `main`. Łączna różnica to 30 plików, 4012 dodanych i 4 usunięte linie. Sam ostatni commit zmienia 16 plików: 1629 dodanych i 13 usuniętych linii. Pierwszy commit SSSHANKS w porównaniu z `main` zmienia dokumentację, ignorowane pliki i zależności; nie należy przypisywać mu autorstwa istniejącego wcześniej całego kodu Python.

Analiza obejmuje wszystkie pliki zmienione w tych czterech commitach, ich połączenia ze wspólnymi modelami, indeksowaniem danych, treningiem i ewaluacją, testy gałęzi oraz zgodność z obecnymi niezacommitowanymi zmianami w `D:/GSN_EEG`. Osobny worktree zawiera dokładny snapshot `89142c2`; kod źródłowy katalogu roboczego użytkownika nie został zmieniony.

## Co rzeczywiście robi projekt

Źródłem wejścia jest THINGS-EEG2: 10 osób, 17 kanałów, 100 próbek czasowych. Modele odcinają pierwsze 20 próbek. Wiele ścieżek uśrednia powtórzenia odpowiedzi na ten sam obraz. To dekodowanie odpowiedzi na znany protokół bodźców, a nie dowolnych myśli.

Istniejący kod semantyki łączy liniowy model zależny od osoby ze wspólną siecią czasowo-przestrzenną i poprawką semantyczną. Przewiduje 384-wymiarowe cechy DINO. Przestrzenny predyktor buduje siatkę DINO 8×8. Starsze modele koloru i krawędzi mają własne enkodery EEG, lecz ich końcowe renderowanie poprawia wynik zamrożonej ścieżki bazowej. Dlatego w tej wersji nie są w pełni niezależne od semantyki.

SSSHANKS dodaje:

1. Opakowania Lightning istniejących modeli, konwersję checkpointów i wspólną konfigurację treningu.
2. Fuzję reprezentacji różnych osób: średnią albo uczone ważenie. Fuzja przyjmuje prawdziwe identyfikatory osób, a trening losuje wielkość podzbioru osób. To fuzja osób, nie połączenie koloru, krawędzi i semantyki w generatorze.
3. Banki cech CLIP, SigLIP2 i EVA02, porównanie regresji ridge, trening enkodera do nowej przestrzeni cech i małą głowicę CLIP.
4. Bank momentów rozkładu VAE i adapter PixArt: CLIP 768 → 8 tokenów → dodatkowa uwaga krzyżowa w blokach zamrożonego transformera. Generator dostaje osobną ścieżkę obrazu z uczoną wagą dla każdego bloku.

Ostatni etap trenuje i ocenia generator na **cechach prawdziwych obrazów**. To sensowny eksperyment izolujący generator, ale nie kompletny test EEG → obraz. Nie ma tu gotowego przebiegu łączącego predykcje EEG, głowicę CLIP i generator z niezależną ewaluacją końcowych rekonstrukcji. Kolor i krawędzie nie warunkują tego adaptera.

## Najważniejsze ustalenia

### 1. P1 — domyślny trening semantyki Lightning nie startuje

Źródła: `eeg_mae/lightning/data.py:76` (snapshot `89142c2`), `eeg_mae/lightning/eeg_encoder.py:50` (snapshot `89142c2`).

`SemanticDataModule` zwraca wiele loaderów w `CombinedLoader(mode="sequential")`. Walidacja nie przyjmuje `dataloader_idx`, a pętla treningowa Lightning nie obsługuje tego trybu CombinedLoader.

Reprodukcja na syntetycznych danych, bez datasetu:

- Domyślny sanity check: `RuntimeError: You provided multiple val_dataloader, but no dataloader_idx argument ...`.
- Po wyłączeniu walidacji: `ValueError: _FitLoop does not support the CombinedLoader(mode="sequential") mode`.

To dotyczy domyślnego `--fusion legacy`. Ścieżka ensemble ma inny loader i nie jest objęta tym konkretnym błędem. Naprawa powinna udostępnić pojedynczy obsługiwany strumień batchy, zachować identyfikatory osób i poprawić agregowanie metryk walidacji. Samo dodanie argumentu do walidacji nie wystarczy.

### 2. P1 przy integracji — obecna wersja lokalna jest niezgodna z opakowaniami SSSHANKS

Źródła: `eeg_mae/lightning/color.py:34` (snapshot `89142c2`), `eeg_mae/lightning/edges.py:32` (snapshot `89142c2`), aktualne pliki w `D:/GSN_EEG/eeg_mae/models` i `training`.

Lokalne zmiany usuwają `residual_smoothness` oraz `residual_energy`, które importuje jego kod. Zmieniają też konstrukcję i wywołania `EEGColorEncoderPipeline`, `EEGEdgeEncoderPipeline` oraz specjalistów: nowe modele przewidują wynik bez przekazywania starego baseline. Opakowania Lightning nadal budują i wywołują starszą wersję pipeline.

Ponieważ `lightning/__init__.py` od razu importuje moduły koloru i krawędzi, niezgodny import może zablokować także korzystanie z innego podmodułu Lightning. Nie jest to błąd samodzielnego snapshotu `s_dev`; to potwierdzona niezgodność dwóch rozwijanych wersji. Potrzebny jest świadomy port opakowań do nowych interfejsów.

### 3. P2 — test szumu nie mierzy deklarowanej odporności na błędy EEG

Źródło: `eeg_mae/evaluation/eval_dit_adapter.py:113` (snapshot `89142c2`), `eeg_mae/evaluation/eval_dit_adapter.py:175` (snapshot `89142c2`).

Opis mówi o szumie odpowiadającym zmierzonemu błędowi enkodera EEG. W implementacji skala to `cond.std()`, czyli rozrzut cech prawdziwych obrazów. Kod nie wczytuje błędów predykcji EEG. Wartość `noise_scale=1` nie oznacza więc rzeczywistego poziomu błędu EEG.

Dodatkowo podczas tej ablacji `metrics(noisy, ...)` porównuje wygenerowane obrazy z zaszumionymi embeddingami. Pole `retrieval_top1` opisane jako wyszukiwanie oryginału korzysta wtedy z zaszumionych warunków zamiast stałej puli oryginałów. To mierzy inne zjawisko niż zachowanie zgodności z oryginalnym obrazem. Obie metryki mogą być przydatne, ale muszą być rozdzielone i poprawnie nazwane.

Należy kalibrować szum na resztach `predicted_clip - true_clip` z walidacji albo użyć rzeczywistych predykcji EEG. Pulę referencyjną do rekonstrukcji trzeba utrzymać stałą. Argument `--official-clip-bank` jest obecnie zadeklarowany, ale niewykorzystany.

### 4. P2 — podgląd generacji zmienia zawartość checkpointu adaptera

Źródło: `eeg_mae/lightning/dit_adapter.py:163` (snapshot `89142c2`), `eeg_mae/lightning/dit_adapter.py:244` (snapshot `89142c2`).

`_vae()` zapisuje model jako `self._vae_model`, przez co PyTorch rejestruje go jako podmoduł. `state_dict()` usuwa tylko klucze `transformer.*`; po pierwszym podglądzie w zapisie pojawiają się wagi `_vae_model.*`. VAE pozostaje także na urządzeniu po podglądzie.

Mała reprodukcja z zastępczym modelem VAE wykazała dodatkowe klucze i błąd ścisłego wczytania do świeżego adaptera: `unexpected=['_vae_model.weight', '_vae_model.bias']`. Skrypty ewaluacji używają `strict=False`, więc ten konkretny błąd mogą ominąć, ale nie usuwa to zbędnych wag z pliku i osłabia kontrolę poprawności ładowania.

Należy zapisywać jawnie tylko projekcję i procesory adaptera oraz utrzymywać VAE poza serializowanym modułem albo wyraźnie filtrować jego stan.

### 5. P2 — `--resume` adaptera przywraca jedynie wagi

Źródło: `eeg_mae/lightning/train_dit_adapter.py:141` (snapshot `89142c2`).

Checkpointy są zapisywane z `save_weights_only=True`, a kod wczytuje model i uruchamia `trainer.fit(...)` bez `ckpt_path`. Stan optymalizatora, postęp treningu i historia wyboru najlepszego modelu nie są kontynuowane. To ponowny trening od zapisanych wag, mimo nazwy i opisu flagi sugerujących wznowienie Lightning. Jest to świadomie opisany kompromis dotyczący wielkości zapisu, ale ma istotne skutki dla interpretacji przebiegów.

Pełne wznowienie wymaga zapisu stanu treningu i `fit(..., ckpt_path=...)`. Alternatywnie należy nazwać tę opcję inicjalizacją z wag i jasno raportować reset stanu.

### 6. P2 — podgląd może generować się wielokrotnie w tym samym kroku

Źródło: `eeg_mae/lightning/train_dit_adapter.py:84` (snapshot `89142c2`).

Callback reaguje po każdym batchu, lecz sprawdza licznik kroków optymalizatora. Przy domyślnym `accumulate=4` licznik pozostaje taki sam przez kilka batchy. Dla `global_step=500` mała reprodukcja wywołała generowanie cztery razy i kierowała zapis do tego samego pliku. DPM sampling jest kosztowny, więc nie jest to tylko kosmetyka logów.

Rozwiązanie: zapamiętać ostatni obsłużony krok albo wywoływać podgląd wyłącznie po nowym kroku optymalizatora.

### 7. P2 — checkpoint ensemble zależy od pliku użytego do inicjalizacji

Źródło: `eeg_mae/lightning/ensemble_trunk.py:40` (snapshot `89142c2`).

Checkpoint ensemble zawiera wytrenowane wagi trunku, ale `load_from_checkpoint` uruchamia konstruktor, który ponownie ładuje `trunk_checkpoint` ze starej ścieżki. Przeniesienie modelu bez tego początkowego pliku powoduje `FileNotFoundError`, zanim właściwe wagi zostaną odtworzone. Zostało to odtworzone na małym zapisanym modelu.

Należy rozdzielić inicjalizację nowego treningu od odtwarzania kompletnego checkpointu. Jawna zależność od osobnego trunku w przekonwertowanych gałęziach koloru i krawędzi jest natomiast zamysłem architektury i nie powinna być automatycznie traktowana jako ten sam błąd.

### 8. P1 dla wiarygodności ewaluacji — starszy błąd przy remisach

Źródło: `eeg_mae/evaluation/official_200way_benchmark.py:200` (snapshot `89142c2`); analogiczny wzór występuje w `eval_dit_adapter.py`.

Ranking jest liczony jako `1 + liczba wyników ściśle większych od wyniku poprawnego obrazu`. Jeżeli wszystkie podobieństwa są równe, każdy poprawny obraz dostaje rangę 1. Dla 200 całkowicie zerowych predykcji funkcja zwróciła `top1=1.0`, `MRR=1.0`, choć nominalny poziom losowy wynosi 0.005.

Ten błąd istniał **przed commitami SSSHANKS**. Jego nowe ewaluatory i trening ensemble korzystają jednak z tej funkcji, a nowa ewaluacja generatora powiela sposób liczenia rang. Trzeba jawnie obsługiwać remisy i zerowe/niepoprawne reprezentacje. To nie dowodzi, że raportowane 54% jest błędne; dowodzi, że metryka nie nadaje się jeszcze do bezwarunkowego wykrywania zapadnięcia modelu.

## Dalsze ograniczenia i dobre elementy

- Testy równoważności sprawdzają wyjścia opakowań modeli przy identycznych wagach. Nie sprawdzają pełnego `Trainer.fit`, wznowienia treningu ani działania generatora. W pliku testów równoważności są trzy testy inferencji; nie ma testów pełnej zgodności dynamiki treningu.
- Lightning koloru i krawędzi aktualizuje EMA, lecz walidacja i publiczne `forward` używają bieżącego specjalisty. W starszych trenerach walidacja i wybór modelu opierały się na EMA oraz porównaniu z baseline i przetasowanymi wynikami. Autor sam opisuje brak przeniesienia tych kryteriów; określenie „behavior-preserving” należy ograniczyć do zweryfikowanych wywołań inferencji.
- Brakuje wersjonowanego buildera `artifacts/dit_adapter/t5_embeddings.pt` i skryptu tworzącego domyślny plik cech głowicy CLIP. Dokumentacja wspomina te artefakty, ale nie daje kompletnego odtworzenia ich w repozytorium. Modele CLIP/PixArt nie mają w wywołaniach przypiętych konkretnych rewizji.
- API `n_max=1` może wylosować niepoprawny zakres `randint(2,2)`. Domyślny `n_max=10` nie jest tym dotknięty. Warto także walidować zakresy liczebności i parametry wejściowe ewaluacji.
- Podział danych jest zorganizowany po obrazach, z powtórzeniami przypisanymi do jednego splitu. Nowy `PairDataModule` wybiera jawnie wiersze train i val, kontrolując zgodność kolejności nazw CLIP/VAE. Nie znalazłem w nim bezpośredniego użycia oficjalnego testu do optymalizacji.
- Uczona fuzja używa identyfikatorów osób i jest testowana pod kątem niezmienniczości na permutację. Naprawia problem pozycyjnego indeksowania w swojej własnej ścieżce; starszy `forward_ensemble` nadal ma ten znany błąd dla niekanonicznych podzbiorów.
- Kierunek eksperymentów jest rozsądny: osobno mierzone EEG → cechy oraz cechy → obraz, regresja ridge jako punkt odniesienia, walidacyjny wybór parametrów ridge, zamrożone modele obrazowe i zerowa inicjalizacja dodatkowego wkładu uwagi. Brakuje jednak końcowej kontroli całego połączenia.
- W dokumentacji zapisano 54% top-1 i 82.5% top-5 dla semantycznego benchmarku 200 kandydatów przy ensemble 10 osób i 80 powtórzeniach. Nie są to wyniki jakości generacji ani pojedynczego pomiaru EEG. Nie odtworzono ich w tym przeglądzie.

## Wykonane sprawdzenia

- Odświeżenie referencji Git i porównanie `main` z `origin/s_dev`.
- Pełny zestaw testów snapshotu: **35 passed in 18.80s**.
- `ruff format --check eeg_mae tests`: **70 files already formatted**.
- Dodatkowy lint nowych modułów wykazał 10 drobnych uwag; nie jest to przyczyna głównych ustaleń. Workflow CI uruchamia kontrolę formatowania, nie ten dodatkowy lint.
- Syntetyczne reprodukcje opisanych błędów loadera, checkpointów, callbacku, `n_max=1` i zerowych predykcji: [review_checks.py](review_checks.py), [wyniki JSON](review-checks.json).
- Środowisko: Python 3.13, PyTorch `2.8.0.dev20250613+cu128`, Lightning 2.6.6, torchmetrics 1.3.2. Testy wykonano na CPU w osobnym venv. Początkowy import torchmetrics 1.9.0 był bardzo wolny w zastanym środowisku; do prób użyto 1.3.2, zgodnego z deklaracją zależności Lightning. Nie jest to ustalenie o błędzie projektu.
- VAE w reprodukcji serializacji był małym modelem zastępczym; sprawdzono rejestrację podmodułu i format zapisu, nie jakość ani pełne działanie PixArt. Nie pobierano dużych wag, nie prowadzono pełnego treningu i nie generowano obrazów.

**Ocena:** wartościowa gałąź badawcza, ale wymaga poprawek uruchamiania, checkpointów i ewaluacji przed potraktowaniem jej jako gotowej ścieżki rekonstrukcji EEG. Integracja z lokalnymi modelami bez baseline wymaga dodatkowego dostosowania interfejsów.
