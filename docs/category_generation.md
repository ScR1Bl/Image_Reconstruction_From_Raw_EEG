# Kategorie i generowanie obiektów z EEG — 10 osób

## Stan i zakres

Działający eksperyment łączy EEG **10 osób × 4 powtórzenia, 0–800 ms**. Nie jest
wynikiem jednej osoby. Kod obejmuje przygotowanie banku treningowego, trening
klasyfikatora w Lightning, przewidywanie kategorii bez obrazu wejściowego i DiT.
Nie uzyskaliśmy jeszcze niezawodnego rozpoznawania 1654 klas ani dowodu odzyskania
orientacji i detali konkretnego egzemplarza. Gotowość kodu nie oznacza osiągnięcia
tego celu badawczego.

Wagi i wyniki pozostają poza Git; praca znajduje się na gałęzi
`research/eeg-dit-ten-subjects`, bez zmiany `main`.

## Co wykazały pomiary

Porównanie klasyfikacji korzysta z **tego samego słownika 1654 kategorii** i 1654
obrazów starej walidacji. Każda kategoria ma 8 innych zdjęć treningowych. Obrazy
walidacyjne są wykluczone z banku referencji. Są to znane kategorie, nie zero-shot.

| Metoda | Top-1 kategorii | Top-5 kategorii |
|---|---:|---:|
| EEG kolegi → prototypy treningowe CLIP | 6,95% | 23,70% |
| Nasz hierarchiczny EEG → prototypy treningowe | 17,59% | 39,72% |
| Uczony klasyfikator z obu reprezentacji EEG | 17,84% | 41,05% |
| Klasyfikator, przemieszane EEG | 0,06% | 0,54% |

Zysk klasyfikatora względem prototypów hierarchicznych jest mały: **0,24 punktu
procentowego** Top-1. Główną poprawę daje użycie reprezentacji hierarchicznej.
Nie ma podstaw do ogłaszania dużego sukcesu dodatkowej głowy klasyfikacyjnej.
Kontrola losowania to jedna permutacja, nie pełny test statystyczny.

Po zamrożeniu głowy wykonaliśmy również ocenę na **1654 innych obrazach
historycznego testu**: Top-1 **15,90%**, Top-5 **38,63%**, przemieszane EEG Top-1
0,12%. Bez kolejnego dopasowania wag. Ten zbiór był wcześniej oceniany dla enkodera
hierarchicznego, więc nie nazywamy go nowym, nietykanym testem całego projektu.
Raport: `runs/dit_ten_subjects/category_classifier/historical_test/report.json`.

Wcześniejsze 74,85% modelu hierarchicznego dotyczyło wyszukiwania prawdziwego obrazu
w galerii zawierającej jego cechy. Nie oznaczało takiej trafności rozpoznawania
kategorii bez dostępu do tego obrazu. Podobnie 14,21% surowego CLIP w poprzednim
raporcie jest innym zadaniem niż powyższe 6,95%; tych liczb nie należy zamieniać.

Klasyfikator ma liniową głowę, standaryzację wyłącznie ze zbioru treningowego,
dropout 0,3 i cross-entropy z label smoothing 0,1. AdamW: lr 0,001, weight decay
0,1; batch 256. Najlepsza była epoka 2 (liczona od zera), trening zatrzymał się po
epoce 14. Enkodery są zamrożone. Checkpoint wybrano przez Top-1 walidacji, bez
wybierania obrazów wygenerowanych. Wagi: `category_classifier/checkpoints/best.ckpt`.

## Dlaczego krawędzie i kolor nie naprawiły obiektu

W [porównaniu mostków do CLIP](dit_ten_subjects.md) MSE i cosinus poprawiły się,
ale rozróżnianie obrazów pogorszyło. W pilocie 8 obrazów mostek S miał 1/8 trafień
Top-1 w galerii prawdziwych obrazów; S+E+C miał 0/8. To mały pilot, ale nie daje
argumentu, by uczynić E/C obowiązkowym warunkiem generacji. Zachowaliśmy je jako
ablację. Globalny mostek nie jest przestrzennym podłączeniem tokenów do DiT.

Sam adapter również ogranicza wynik: nawet kontrola podająca prawdziwy CLIP obrazu
myli detale i niektóre przedmioty. Lepszy enkoder EEG nie usuwa automatycznie tego
błędu generatora.

DINO może być lepszym celem dekodowania pewnych cech EEG, ale porównanie cosinusów
0,1038 i 0,0887 w **różnych przestrzeniach** nie dowodzi odzyskania 17% jakości
rekonstrukcji. Adapter przyjmuje rozkład cech, na którym był trenowany; możliwość
nauczenia nowych K/V nie oznacza, że obecne wyuczone K/V obsługują dowolny zamiennik.
Dlatego w gotowym torze rozpoznawanie korzysta z reprezentacji hierarchicznej i EEG
CLIP, a istniejący adapter nadal dostaje właściwy CLIP obrazu treningowego. DINO
można oceniać osobno w równym protokole i uczyć dla niego własne warunkowanie.

## Trzy jawne tryby

1. `prior`: klasyfikator przewiduje kategorię; model hierarchiczny wybiera najbliższe
   zdjęcie **treningowe tej przewidzianej kategorii**. Jego CLIP trafia do adaptera.
2. `text`: przewidziana nazwa kategorii trafia przez T5 do PixArt. Adapter obrazowy
   jest wyłączony. To wizualizacja hipotezy kategorii; nie zachowuje geometrii EEG.
3. `text-and-prior`: nazwa przewidzianej kategorii i treningowy CLIP są podawane
   wspólnie. Daje możliwość sterowania semantyką, ale może dziedziczyć układ zdjęcia
   treningowego i błędy adaptera.

Prawdziwa kategoria bodźca nigdy nie wybiera referencji ani promptu. Raport podaje
Top-5 i użyte zdjęcie treningowe. Wynik Top-1 jest hipotezą; margines logitów **nie
jest skalibrowanym prawdopodobieństwem**. Fotorealizm nie dowodzi poprawnej kategorii.

Bank tekstowy zawiera wyłącznie nazwy klas, bez opisu pozy, koloru czy tła bodźca.
T5-XXL jest zamrożony, używa NF4 i obliczeń bfloat16; embeddingi zapisujemy wcześniej,
a T5 zwalniamy przed uruchomieniem DiT. Kwantyzacja jest dodatkowym przybliżeniem.

## Uruchomienie

Z katalogu głównego repozytorium, po zainstalowaniu zależności:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[training,visual-teachers,category-text,dev]"
.venv/Scripts/python.exe -m eeg_mae.evaluation.reconstruct_eeg --eeg runs/dit_ten_subjects/category_classifier/demo_eeg.npy --output runs/moj_wynik --mode prior
```

`--eeg` to plik NumPy `[N,10,4,17,100]` lub `[10,4,17,100]`. Wymaga **dokładnie
preprocessingu i kolejności kanałów z cache tego projektu**, osób sub-01…sub-10
oraz 4 powtórzeń. Wejście zachowuje pełne 100 próbek cache, wraz z pierwszymi 20
próbkami przed bodźcem; enkoder sam odcina te 20 i używa 80 próbek po bodźcu
(0–800 ms przy 100 Hz). Nie należy rozciągać 800 ms do 100 próbek. Nie przyjmuje
surowych napięć z dowolnego urządzenia. Przykład
`demo_eeg.npy` zawiera pierwsze 8 obrazów walidacji, nie dane nowych osób.

Opcje `--mode text` i `--mode text-and-prior` wymagają banku `category_text.pt`.
`--no-render` wykonuje samo rozpoznawanie. Katalog wyjściowy musi być pusty.
Wynik zawiera PNG, `conditions.pt` i `report.json` z predykcjami, parametrami,
hashami wejść i statusem ukończenia. Domyślne wagi wskazują lokalne artefakty.

Odtworzenie treningu w nowych katalogach wymaga przekazania zgodnych `--prior`,
`--output`, `--classifier` i `--bank`; nie nadpisujemy ukończonych eksperymentów:

```powershell
.venv/Scripts/python.exe -m eeg_mae.data.build_category_prior --output runs/nowy_prior
.venv/Scripts/python.exe -m eeg_mae.lightning.category_classifier --prior runs/nowy_prior --output runs/nowy_klasyfikator
.venv/Scripts/python.exe -m eeg_mae.data.build_category_text --bank runs/nowy_prior/training_bank.pt --output runs/nowy_text.pt
```

Przerwany trening głowy można wznowić przez `--resume`. Ukończony trening jest
chroniony przed przypadkowym nadpisaniem. Renderer porównań sprawdza zgodność
checkpointu, warunków i parametrów przy wznowieniu.

## Ograniczenia i warunek wyniku badawczego

To stara walidacja, używana wcześniej do wyboru enkoderów i teraz do wyboru głowy.
Predykcje enkoderów na treningu są in-sample. Nie przedstawiamy tej oceny jako nowego,
nietykanego testu; historyczny test enkodera hierarchicznego też już był oglądany.
Nie potwierdzono rozłączności sekwencji prezentacji. Nowy protokół jednej osoby
pozostaje osobnym eksperymentem, bez otwierania jego testu.

Aby twierdzić „ten samolot skierowany w tę stronę”, trzeba wykazać rozróżnianie
orientacji i układu części pomiędzy obrazami **tej samej kategorii**, na niezależnym
protokole, oraz z kontrolą zamiany EEG. W tym eksperymencie jeszcze tego nie ma.

## Powiązane rozwiązania

[ATM / Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://arxiv.org/abs/2403.07721)
rozdziela dekodowanie EEG i generowanie sterowane cechami. [IP-Adapter](https://arxiv.org/abs/2308.06721)
opisuje oddzielne attention dla obrazu i tekstu oraz ich wspólne użycie. To uzasadnia
sprawdzenie obu kanałów generatora; nie gwarantuje wyników naszych wag. Nasz wariant
z treningowym obrazem jest generowaniem wspomaganym wyszukiwaniem, nie nowym dowodem
bezpośredniego odzyskiwania struktury z EEG.
