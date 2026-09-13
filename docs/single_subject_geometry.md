# Geometria obrazu z EEG jednej osoby: implementacja i eksperymenty

Stan: 13 września 2026. Branch `research/eeg-single-subject-geometry`; `main` bez zmian.
Kolejka diagnostyczna zakończyła się bez błędów o 01:44 czasu polskiego.
To zakończony etap diagnostyczny, a nie ukończona rekonstrukcja EEG → PixArt.
Pierwotny [plan badawczy](experiments/plan_geometry_v2.md) pozostaje archiwum założeń.
Zachowano też [pełny przegląd commitu SSSHANKS](experiments/REVIEW_SSSHANKS.md)
i [wcześniejszą analizę struktur, nauczycieli i publikacji](experiments/ANALIZA_EEG_STRUKTURA_DINO_CLIP.md).
To dokumenty sprzed implementacji; bieżące wyniki i naprawy opisuje niniejszy plik.

## Co już działa

- Zachowano wcześniejsze lokalne zmiany użytkownika w `68cfdc3`.
- Dodano osobny protokół jednej osoby i treningi Lightning (`c6d4672`). Z kodu SSSHANKS,
  commit `89142c2` na `origin/s_dev`, przeniesiono wybrane zgodne moduły: `cli.py`,
  `common.py`, `subject_fusion.py`, `dit_adapter.py`. Nie scalono jego niezgodnych
  loaderów i starych wrapperów koloru/krawędzi. Nowe treningi korzystają z obecnych,
  niezależnych modeli projektu.
- Zakończono ridge dla DINO, CLIP-B/32, SynCLR i maski, w oknach 200/400/800 ms.
- Zakończono osobne treningi neuronowe DINO, CLIP-B/32, SynCLR, krawędzi i koloru, 400 ms.
- Zakończono kontrolę dodatkowej geometrii z trzykrotnym cross-fittingiem ridge.
- Zakończono kolejkę trzech foldów najlepszego modelu semantycznego, koloru,
  foldów krawędzi/koloru i końcowych kontroli S / S+E / S+C / S+E+C.
  Faktyczny, aktualizowany stan: `runs/single_subject_geometry/sub-01/pipeline_status.json`.
  Stan kolejki: `complete`, bez zgłoszonych błędów; wszystkie zaplanowane etapy mają raporty.

## Wynik zakończonych kontroli neuronowych

Walidacja sub-01, cztery prezentacje, 0–400 ms, te same 1179 par kategorii.
Mały model liniowy przewiduje maskę z trzykrotnie cross-fitowanych przewidywań
EEG. S oznacza SynCLR, E maskę16, C przewidywany RGB zmniejszony do 8×8.

| Wejście do modelu maski | 2AFC geometrii | Zysk nad S | Eksploracyjny przedział 95% zysku |
|---|---:|---:|---:|
| S | 60,90% | — | — |
| S+E | 61,83% | +0,93 pp | +0,08 do +1,74 pp |
| S+C | 61,32% | +0,42 pp | −0,13 do +0,98 pp |
| S+E+C | 61,96% | +1,06 pp | +0,21 do +1,91 pp |
| S+losowe cechy | 61,07% | +0,17 pp | −0,72 do +1,06 pp |

Próg +5 pp nie został osiągnięty. Krawędzie dostarczają małą dodatkową informację
w tej kontroli; osobna korzyść koloru dla geometrii pozostaje nieprzekonująca.
To nie dowodzi bezużyteczności koloru dla palety ani redundancji pełnych tokenów:
kontrola korzysta z dekodowanych map, a celem jest maska. Przedziały są walidacyjne,
po wyborze hiperparametrów; nie uwzględniają wielokrotnego testowania ani zmienności
ziaren/osób i nie są końcowym potwierdzeniem badawczym.

Zamiana E/C między dwoma obrazami tej samej kategorii obniża S+E+C do 55,94%.
Poprawne przypisanie struktury pomaga więc temu modelowi, ale spadku po celowym
podaniu błędnych warunków (+6,02 pp względem kontroli) nie wolno przedstawiać jako
zysku nad samą semantyką (+1,06 pp).

Bezpośrednia maska encodera krawędzi ma nadal 62,98% i przewyższa powyższe mapowanie
S+E+C w 2AFC. Mapowanie S+E+C ma za to niższy MSE: 0,1427 vs 0,1596. Dobierano je
według MSE; obecne połączenie nie jest najlepsze we wszystkich kryteriach.
Wszystkie wybrane alfy ridge znalazły się na górnej granicy sprawdzanej siatki
(10000). Warto zbadać większą regularyzację na walidacji, lecz nie zmieniać po fakcie
znaczenia tego zakończonego eksperymentu.

Gałąź koloru: rozróżnianie chromy dwóch zdjęć tej samej kategorii wynosi 54,99%
przy czterech prezentacjach, 53,69% przy dwóch i 52,66% przy jednej (losowo 50%).
To osobna miara, z 1654 parami; nie porównujemy jej bezpośrednio z 2AFC maski.
Odświeżony `diagnostics/validation_pairs.png` zawiera piąty rząd z przewidywanym
kolorem. Pokazuje bardzo rozmyte układy barw, nie wierne rekonstrukcje obiektów.

Wniosek: nie uruchamiamy na podstawie tego wyniku kosztownego treningu dyfuzji.
Najpierw sprawdzić regularyzację mapowania, informację w pełnych tokenach i cele
orientacji/części oraz audyt czasu. Nie ma jeszcze dowodu odzyskania układu skrzydeł.
Test pozostaje zamknięty. [Pełny raport kontroli](experiments/sub01_neural_geometry_20260913.json)
zapisuje wszystkie warianty, przedziały i status progu.

## Protokół i znaczenie wyników

Badany `sub-01` (`--subject 0`), bez EEG innych osób na treningu ani inferencji.
Model uczy się ze średniej czterech prezentacji; ewaluacja osobno dla pierwszych 1, 2 i 4.
Jedna osoba nie oznacza pojedynczej prezentacji. Powtórzenia tego samego obrazu zawsze
pozostają razem. Obecne metryki nie uśredniają różnych losowych podzbiorów powtórzeń.

Podział: 6/2/2 obrazów w każdej z 1654 kategorii, czyli 9924 treningowych, 3308
walidacyjnych, 3308 testowych. To nowe egzemplarze znanych kategorii. Oficjalne 200
nieznanych kategorii THINGS-EEG2 i nowy wewnętrzny test nie były oceniane.
Maski ocenia się tylko dla poprawnych celów nauczyciela; porównanie wewnątrz kategorii
obejmuje 1179 par, w których obie maski są dostępne.

Zamrożony protokół: `data/derived/single_subject_geometry/protocol.json`, SHA-256:
`375519f696a51fdcce4333cff27a115ae2938c228e82baf9825e3b6aee3a5fdf`.
Starych checkpointów EEG nie użyto do inicjalizacji nowego podziału: mogły widzieć
obrazy nowego testu. Semantyka startuje od ridge wyuczonego wyłącznie na nowym train;
w foldach ridge jest dopasowywany ponownie bez odłożonych obrazów.

**Podział jest rozłączny po obrazach, ale nie ma jeszcze potwierdzonej rozłączności
sekwencji prezentacji.** Lokalne EEG ma czasy od −200 do +790 ms; indeks 20 odpowiada
0 ms. THINGS-EEG2 prezentuje bodźce przez 100 ms, z SOA 200 ms; późna część okna
może obejmować odpowiedzi na następne bodźce. Lokalne banki nie zawierają oryginalnych
zdarzeń sesji. Raporty mają `sequence_disjoint_verified: false`. Nie stwierdzono
tym samym przecieku; jego wykluczenie wymaga metadanych prezentacji i audytu
preprocessingu. Źródło protokołu bodźców: [Gifford i in.](https://pubmed.ncbi.nlm.nih.gov/36400378/).

## Zakończone porównanie semantyki

Walidacja, 4 prezentacje, 400 ms. Każdy model dopasowuje EEG do własnej zamrożonej
przestrzeni wizualnej. Identyfikacja obrazu ma 3308 kandydatów; kategoria to etykieta
najbliższego obrazu w tej galerii, a nie osobny klasyfikator tekstowy.

| Cel | Top-1 obrazu | MRR | Trafna kategoria najbliższego obrazu | 2AFC w tej samej kategorii |
|---|---:|---:|---:|---:|
| DINOv2-S, 384d | 2,96% | 0,0693 | 3,75% | 71,19% |
| CLIP-B/32, 512d | 3,39% | 0,0782 | 4,17% | 73,88% |
| SynCLR, 768d | 4,87% | 0,1061 | 5,83% | 77,99% |

Losowe Top-1 obrazu: około 0,030%; kategorii: 0,060%; 2AFC: 50%.
2AFC w przestrzeni semantycznej może wykorzystywać kolor lub tło, więc nie jest
równoważne rozpoznaniu orientacji. MRR służył wyborowi checkpointów; dodatkowe
metryki kategorii policzono po treningu, bez zmiany selekcji.

W tym pilotażu do kontroli wybrano SynCLR. Liniowy ridge preferował DINO nad CLIP,
ale po treningu neuronowym kolejność się odwróciła. Nie jest to porównanie z CLIP-L/14
768d kolegi ani dowód uniwersalnej przewagi jednej rodziny. Liczb 0,1038 i 0,0887
nie udało się przypisać do dostępnego checkpointu i zgodnego protokołu. Ich iloraz
nie oznacza odzyskania 17% jakości obrazów.

## Geometria: sygnał istnieje, dokładność pozostaje ograniczona

| Model, 400 ms | Prezentacje | 2AFC maski w tej samej kategorii | IoU maski | Błąd osi |
|---|---:|---:|---:|---:|
| Ridge EEG → maska | 4 | 62,64% | 0,4931 | 27,70° |
| Neuronowy encoder krawędzi | 1 | 56,19% | 0,4618 | 37,67° |
| Neuronowy encoder krawędzi | 2 | 58,74% | 0,4826 | 33,78° |
| Neuronowy encoder krawędzi | 4 | 62,98% | 0,5060 | 31,19° |
| Ten sam model, losowo zamienione EEG | 4 | 50,08% | 0,4352 | 44,29° |

Oś jest bezkierunkowa, modulo 180°; nie mówi, gdzie samolot ma nos. Oceniana maska
16×16 pochodzi z nauczyciela wizualnego. To nie ręczna adnotacja skrzydeł, pozy ani
orientacji przód/tył. Panel o stałym ziarnie (`diagnostics/validation_pairs.png`)
pokazuje rozmyte, ogólne maski. Wynik ponad przypadek nie uprawnia do twierdzenia,
że odzyskaliśmy wierny układ części. Duża sieć nie wyprzedza ridge we wszystkich miarach.

Kontrola ridge używała przewidywań EEG poza własnym foldem treningowym do nauki
mapowania semantyka → maska. Dodanie osobnej predykcji maski zwiększało 2AFC o:

| Semantyka | Zysk S+E nad S | Przedział bootstrap 95%, kategorie |
|---|---:|---:|
| DINO | +1,10 pp | +0,25 do +1,95 pp |
| CLIP-B/32 | +0,93 pp | +0,08 do +1,78 pp |
| SynCLR | +2,37 pp | +1,57 do +3,27 pp |

To eksploracyjne przedziały na walidacji po doborze hiperparametrów, nie końcowe
potwierdzenie statystyczne. E oznacza tutaj ridge do maski, nie neuronowe tokeny
krawędzi. Mała dodatkowa informacja uzasadnia kontrolę neuronową, ale jeszcze nie
próg +5 pp planowany dla kosztownego etapu generacyjnego.

## Co dokładnie robi uruchomiona kolejka

`python -m eeg_mae.training.run_single_subject_pipeline` wykonuje zadania kolejno,
na jednym GPU. Każdy bieg ma maksimum 30 epok dla struktury/koloru, 35 dla semantyki,
early stopping po 7 epokach bez poprawy i limit procesu 2 godziny. Błąd zatrzymuje
całą kolejkę. Nie uruchamia automatycznie testu, pobierania modeli ani dyfuzji.

1. Wybiera semantykę według zakończonego MRR walidacyjnego (obecnie SynCLR).
2. Trenuje trzy modele z odłożeniem po dwóch z sześciu obrazów treningowych kategorii.
   Zapisuje przewidywania odłożonych obrazów oraz walidacji.
3. Sprawdza pełny model koloru na 128 obrazach przez jedną epokę; następnie trenuje
   go na całym train. Nie używa starego, lokalnie trenowanego oracle koloru.
4. Trenuje trzy foldy krawędzi i trzy foldy koloru.
5. Dopasowuje małe mapowania do maski z przewidywań S, S+E, S+C i S+E+C.
   E to przewidywana maska16, C to przewidywany RGB zmniejszony do 8×8. To kontrola
   odzyskanej informacji, jeszcze nie pełnych latentnych tokenów.
6. Dodaje kontrolę z losowymi cechami o tej samej liczbie współczynników co S+E+C
   i zamianę E/C między dwoma obrazami tej samej kategorii przy zachowanej semantyce.
   Zapisuje przedziały zysku oraz informację, czy został osiągnięty próg +5 pp.

Modele z foldów widzą mniej obrazów niż model pełny, więc pozostaje różnica rozkładu
predykcji OOF i pełnych. S vs S+E ma różną liczbę współczynników; kontrola S+szum
pomaga zbadać sam wzrost wymiaru, lecz nie zastępuje późniejszego eksperymentu
z identycznym budżetem parametrów generatora. Wszystkie selekcje są walidacyjne.

## Lightning, checkpointy i różnice względem starych trenerów

Nowy `SingleSubjectLightning` trenuje obecne niezależne encodery, `subjects=1`.
EMA aktualizuje się po kroku optymalizatora i służy walidacji. Pełne checkpointy
zawierają optimizer, scheduler, stan AMP i RNG; wznowienie sprawdza protokół, osobę,
okno i kluczowe ustawienia. Test CPU potwierdza dokładną zgodność wag, EMA i stanu
AdamW: dwie epoki ciągiem vs jedna epoka i wznowienie. Nie jest to obietnica bitowej
deterministyczności każdego kernela CUDA ani wznowienia w środku batcha.

Nowe biegi zapisują wersje środowiska, Git oraz hash źródeł. Cztery pierwsze pełne
biegi powstały z implementacji `c6d4672` przed dodaniem tych pól. Kod oceny i kolejki
dodano w `0b27a77` i `a2d5411`.

Semantyka: symetryczny InfoNCE + strata cosinusowa. Krawędzie: cele piramidy,
CVaR, ranking z błędnymi parami, topologia i różnorodność; bez regularizatora
wymagającego co najmniej trzech osób i bez części dawnych strat spread/variance.
Kolor: bezpośredni RGB/OKLab, deskryptor i rzadka chroma; bez dystylacji ze starego
lokalnego oracle. To nowe warianty, nie identyczne odtworzenie dawnych biegów.

Naprawiono też gradientową stratę `FusionPrior` i zapis adaptera PixArt tak, aby
nie dołączał leniwie załadowanego, zamrożonego VAE. Test serialization nie pobiera
modelu PixArt. Cały zestaw po dodaniu kontroli: **57 testów przeszło**.

## Dalsze kroki i brakujące elementy

Po zakończeniu kolejki ocenić `neural_complementarity/report.json`, panel i kontrole.
Nie wybierać wygodniejszej metryki po zobaczeniu testu. Jeśli geometria nie daje
istotnej dodatkowej informacji, najpierw poprawić jej cele/enkoder albo usunąć
zbędną gałąź. Jeśli daje — przygotować przestrzenne warunkowanie zamrożonego
generatora i kontrolę tej samej pojemności, dopiero potem zamrożony test.

Potrzebne przed odtworzeniem konkretnego wyniku SSSHANKS: checkpoint biegu A,
wytrenowany adapter PixArt, jego konfiguracja i `t5_embeddings.pt`. Użytkownik
zapytał autora; plików na razie nie ma. Ewaluator PixArt z jego commitu podaje
cechy prawdziwego obrazu, a nie predykcję EEG. Adapter tworzy K/V jako kopie wag
tekstowych; projektor jest nowy. Osiem tokenów z wektora globalnego nie stanowi
siatki przestrzennej. Nie uruchomiono udawanego odtworzenia tego wyniku.

Przed mocnym wynikiem badawczym potrzebne są także: audyt sekwencji prezentacji,
adnotacje kierunku/relacji części, powtórzenia ziaren i kolejni badani oceniani
indywidualnie. Nie zakładamy potwierdzonej nowości architektury. Bliskie kierunki
obejmują [BrainHiVE](https://arxiv.org/html/2602.07495v1),
[CogCapPro](https://arxiv.org/html/2603.12722v1) i
[SGDM](https://arxiv.org/html/2604.22649v1); dwa ostatnie wskazane materiały są
preprintami. Porównania nauczycieli również wymagają zgodnego protokołu, zob.
[Rajabi i in., ICML 2025](https://proceedings.mlr.press/v267/rajabi25a.html).

## Odtwarzanie i monitoring

Środowisko lokalne: Python 3.13.14, Torch `2.8.0.dev20250613+cu128`, Lightning 2.6.6,
torchmetrics 1.3.2, RTX 3060 Laptop 6 GB. Używana `.venv` dziedziczy systemowy Torch;
to działające środowisko maszyny, nie pełny przenośny lockfile CUDA.

```powershell
.venv/Scripts/python.exe -m eeg_mae.data.single_subject
.venv/Scripts/python.exe -m eeg_mae.training.train_geometry_probes --threads 2
.venv/Scripts/python.exe -m eeg_mae.lightning.train_single_subject --task synclr --end-ms 400 --epochs 35 --batch-size 256 --ridge-alpha 1000 --output runs/nowy_bieg
Get-Content runs/single_subject_geometry/sub-01/pipeline_status.json
Get-Content runs/single_subject_geometry/sub-01/queue_logs/color_full.log -Tail 5
.venv/Scripts/python.exe -m eeg_mae.evaluation.report_single_subject --panel
```

Nie uruchamiać drugiej kolejki równolegle; `pipeline.lock` chroni przed duplikacją.
Zakończone raporty są wykorzystywane ponownie. Po awarii częściowy bieg wymaga
świadomego `--resume .../checkpoints/last.ckpt` z taką samą konfiguracją i katalogiem.
Przy prawidłowym zakończeniu procesu kolejki blokada jest usuwana; po wymuszonym
zakończeniu należy najpierw sprawdzić, czy proces nadal działa.

Wyniki i wagi: `runs/single_subject_geometry/sub-01/`; banki: `data/derived/`.
Żadne dane ani checkpointy nie są śledzone przez Git. Mały
[snapshot metryk](experiments/sub01_geometry_20260912.json) dokumentuje zakończoną
część eksperymentu; bieżący stan kolejki zawsze pochodzi z pliku statusu.
