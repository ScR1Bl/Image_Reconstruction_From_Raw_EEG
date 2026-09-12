# Plan: geometria konkretnego obrazu z EEG jednej osoby

Wersja 2 — 12 września 2026. Dokument planistyczny; bez implementacji i treningów. Wersja 2 dodaje diagnostykę przed dużymi treningami, audyt czasu EEG i precyzuje znaczenie obecnych celów struktury.

## Cel i protokół

Wykazać, że sygnał EEG jednej osoby pozwala odtworzyć położenie, orientację i wybrane relacje części konkretnego bodźca lepiej niż sama informacja semantyczna. Pierwszy wynik musi dotyczyć jednej osoby. Nie korzystamy z fuzji EEG różnych osób na wejściu.

Pierwszy eksperyment inżynierski: pierwszy badany według identyfikatora datasetu, wybrany przed oceną jakości. Czysty wariant uczymy na jego EEG. Uśrednianie powtórzeń tego samego bodźca u tej osoby jest jawnie oznaczone: podstawowy punkt startowy to cztery powtórzenia dostępne w części treningowej THINGS-EEG2; osobno raportujemy 1, 2 i 4 powtórzenia. Nie przedstawiamy czterech powtórzeń jako pojedynczej próby. Ewentualny trening wspólny na wielu osobach z inferencją jednej osoby byłby oddzielnym, opisanym wariantem.

Po pilotażu architekturę sprawdzamy u dwóch kolejnych, z góry ustalonych osób. Docelowo wynik badawczy powinien obejmować wszystkie dostępne osoby, zawsze z oddzielnymi predykcjami. Średnia metryk osób nie jest fuzją ich EEG.

Nie obiecujemy dokładnego odzyskania wszystkich skrzydeł i części. Pierwszy próg dotyczy położenia i orientacji, następny konfiguracji części tam, gdzie bodźce i adnotacje pozwalają ją ocenić.

## Co wykorzystujemy z obecnego projektu

| Element | Zastosowanie w planie | Niezbędna adaptacja |
|---|---|---|
| Encoder semantyczny DINO i ścieżka CLIP | Dwa kontrolowane cele EEG | Ten sam podział i protokół jednej osoby; bez porównywania niezgodnych biegów |
| Niezależny encoder krawędzi | Kandydat do dodatkowej geometrii | `subjects=1`, właściwy loader, ponowny trening |
| Niezależny encoder koloru | Kandydat do palety i rozmieszczenia chromy | `subjects=1`, właściwy loader, ponowny trening |
| Banki cech i celów wizualnych | Przyspieszenie uczenia i powtarzalne cele | Audyt pochodzenia, dopasowania indeksów i uczonych transformacji |
| Lightning kolegi | Wspólny trening, checkpointy i konfiguracja | Naprawa stwierdzonych błędów; zgodność z nowymi modelami |
| PixArt i adapter z `89142c2` | Punkt odniesienia generacji i pierwszy kandydat do integracji | Rozdzielenie wejść oracle/EEG, przestrzenna ścieżka warunkowania |
| Hierarchiczna semantyka SynCLR/CLIP/VAE | Dodatkowy punkt odniesienia po pilotażu | Nie traktować jej uczonej przestrzeni jak natywnego CLIP |
| Własny `FusionPrior` | Późniejszy eksperyment | Nie dokładać od razu jako kolejnego obowiązkowego modułu |

Obecne encodery koloru i krawędzi mają parametr liczby osób, ale trenery i bank warunków zbierają obecnie wiele osób. Samo ustawienie parametru bez zmiany danych nie wystarczy. Nie duplikujemy EEG jednej osoby na dziesięć wejść i nie zerujemy pozostałych dziewięciu, aby nazwać stary model jednoosobowym. Regularizator krawędzi preferujący efektywnie co najmniej trzy osoby trzeba wyłączyć w nowym protokole.

## Etap poprzedzający implementację: czy mierzymy właściwą informację?

Najpierw przygotować tani zestaw diagnostyczny na walidacji, korzystając z istniejących obrazów i dostępnego trybu predykcji semantyki jednej osoby. Nie wymaga to od razu ponownego treningu całej architektury. Stare wieloosobowe gałęzie koloru/krawędzi nie dostarczają jeszcze wyniku tego protokołu.

Porównać ten sam bodziec z kontrolami zmieniającymi jeden czynnik: geometrię, kolor albo tło. Edytowane obrazy są kandydatami do oceny istniejącego EEG, nie nowymi dodatnimi parami treningowymi. Nie mamy EEG odpowiedzi na edytowany bodziec. Kontrole powinny zachowywać pozostałe aspekty oraz obejmować identyczne przetwarzanie bez zamierzonej zmiany, aby oddzielić geometrię od artefaktów edycji.

Jest już bardzo bliski benchmark: [EEG-EditBench, preprint 30.07.2026](https://arxiv.org/html/2607.27857v1), badający rozróżnianie bodźca od kontrolowanych edycji przy stałym EEG. Pokazuje ograniczenia ogólnego retrieval przy rozpoznawaniu atrybutów; zawiera również kontrole dopasowania EEG i wpływu regeneracji obrazu. Publiczne obrazy tego benchmarku pochodzą z oficjalnego testu 200, więc nie należy używać ich do strojenia naszej metody. Analogiczny zestaw rozwojowy przygotować z walidacji, publiczny pozostawić do końcowej oceny. Benchmark jest pomocniczą diagnostyką; nie zastępuje testu na wielu rzeczywistych egzemplarzach z ich własnym EEG.

Równolegle wykonać proste próby: EEG → pozycja/obszar obiektu, EEG → oś główna, EEG → niskowymiarowa reprezentacja obrysu. Punkty odniesienia: średnia treningowa, geometria przewidywana z S, ridge z EEG oraz właściwa gałąź E po adaptacji do jednej osoby. Uczone redukcje wymiaru i normalizacje dopasować tylko na treningu. Te próby pozwalają sprawdzić wartość sygnału przed uruchamianiem generatora.

Dla nauczycieli wizualnych sprawdzić dodatkowo, jak dobrze porównywalny mały dekoder przewiduje geometrię z prawdziwego CLIP/DINO globalnego i cech lokalnych. Nie nazywać niepowodzenia jednego prostego dekodera dowodem całkowitego braku informacji w embeddingu. Taki test mierzy dostępność geometrii dla danego dekodera, a nie absolutną granicę informacji.

**Najbliższy kamień milowy:** raport jednej osoby pokazujący osobno sygnał semantyki i geometrii, jeszcze bez generowanych obrazów. Jego wynik ustala, które gałęzie oraz cele warto dalej rozwijać.

## Etap 0. Wspólna, odtwarzalna baza

Przyszła gałąź robocza: `research/eeg-single-subject-geometry`. Przed integracją zachować dokładny snapshot niezacommitowanych zmian w `D:/GSN_EEG`, wraz z plikami untracked. Uwzględnić wspólnego przodka i commit `89142c2`; włączyć kompatybilne fragmenty kodu kolegi bez zastępowania obecnych niezależnych encoderów starszymi wersjami.

Oddzielić commity: zachowanie aktualnego stanu, naprawy infrastruktury, protokół danych, wariant jednej osoby, eksperymenty reprezentacji, warunkowanie przestrzenne. Nie tworzyć jednego commitu mieszającego refaktor i zmianę modelu.

Zakres techniczny: domyślny przebieg loaderów Lightning, zgodność importów, EMA, pełny resume z optimizerem/schedulerem/RNG, checkpoint bez przypadkowego VAE i ukrytej zależności od pierwotnej ścieżki inicjalizacji, podgląd raz na krok optymalizatora. Metryka retrieval musi poprawnie obsługiwać remisy i stałe embeddingi. Stan początkowy oraz naprawy są opisane w [wcześniejszym przeglądzie](REVIEW_SSSHANKS.md).

**Gotowe, gdy:** przechodzi mały trening, zapis/odczyt/wznowienie zachowują stan, a refaktor istniejącego trybu zachowuje forward, loss i krok optymalizatora w uzgodnionej tolerancji. Osobne testy sprawdzają indeks bodźca i identyfikator osoby.

## Etap 1. Zablokowany test geometrii

Z dziesięciu zdjęć każdej z 1654 kategorii proponujemy podział 6 trening / 2 walidacja / 2 test, z ziarnem i manifestem zapisanymi przed treningiem. To kandydat wymagający poniższego audytu czasu, a nie gotowy dowód niezależności danych. Daje nominalnie 9924 / 3308 / 3308 obrazów; faktyczna liczba po jawnych filtrach jakości będzie podana osobno. Wszystkie powtórzenia EEG jednego obrazu należą do tego samego splitu. Podział obrazów jest wspólny dla wszystkich osób.

Dotychczasowy benchmark 8/1/1 zachować do diagnostyki historycznej. Oficjalne 200 niewidzianych kategorii pozostawić jako osobny test generalizacji, z jawną liczbą uśrednionych powtórzeń. Nowy 6/2/2 mierzy generalizację do nowych egzemplarzy znanych kategorii, a nie zero-shot kategorii.

**Krytyczne:** checkpoint uczony na starych ośmiu zdjęciach może już znać obrazy nowego testu. Nie wolno użyć go jako inicjalizacji i ogłosić testu niezależnym. Ograniczenie obejmuje EEG encoder, adapter uczony na obrazach, lokalnie uczone renderery/oracle, projekcje i normalizacje dopasowane do danych. Uniwersalne zewnętrzne modele wizualne można zamrozić jako nauczycieli; ich pochodzenie i znany zakres pretreningu trzeba opisać.

Do badania struktury utworzyć stały zestaw adnotacji, niezależnie od wygenerowanych wyników: pozycja dominującego obiektu, obrys, oś główna i proporcje. Dla odpowiednich klas dodać przód/tył oraz punkty charakterystyczne części. Oś główna ma niejednoznaczność 180° i sama nie potwierdza kierunku nosa samolotu. Symetryczne lub niejednoznaczne przypadki oznaczyć, a nie usuwać po zobaczeniu wyników modelu.

Nie obracać ani nie odbijać obrazu treningowego przy pozostawieniu niezmienionego EEG: taki target opisywałby bodziec, którego badany nie widział. Augmentacje muszą zachowywać zgodność bodźca z sygnałem.

**Gotowe, gdy:** manifest, reguły adnotacji, audyt przecieku, lista osób i główna metryka są zamrożone. Wyniki starego podziału są wyraźnie oddzielone.

### Audyt czasu i sąsiednich bodźców

Bezpośredni odczyt metadanych `sub-01/preprocessed_eeg_test.npy` potwierdził: tensor 200×80×17×100, oś czasu −0,20…0,79 s, krok 0,01 s. Obecne encodery odcinają pierwsze 20 próbek i korzystają z 0…0,79 s. Opis oryginalnego badania podaje obraz przez 100 ms i początek kolejnego bodźca co 200 ms. [Gifford i in., 2022](https://pubmed.ncbi.nlm.nih.gov/36400378/).

Wniosek metodologiczny: późniejsza część okna może zawierać odpowiedzi także na sąsiednie obrazy. Nie oznacza to automatycznie błędnych wyników — kolejność bodźców i uśrednianie mają znaczenie. Nie można jednak traktować całych 800 ms jako izolowanej odpowiedzi na jeden obraz ani samego splitu po nazwie obrazu jako wystarczającego audytu niezależności czasowej.

Sprawdzić metadane oryginalnych sekwencji i sesji oraz współdzielenie surowych fragmentów czasu pomiędzy epokami różnych splitów. Lokalny indeks ma identyfikatory obrazu i powtórzenia, lecz nie ma kolumn sekwencji/sesji; odczytany plik preprocessed zawiera tylko EEG, nazwy kanałów i oś czasu. Te pliki nie wystarczają do odtworzenia kolejności prezentacji. Nie stwierdzono konkretnego przecieku — wymaga on osobnego audytu zdarzeń.

Jeżeli okna nachodzą między splitami, rozważyć podział sesjami/sekwencjami i wykluczanie nakładających się fragmentów. Jednoczesne zachowanie rozłączności obrazów, grup czasowych oraz liczby powtórzeń może zmniejszyć zbiór; podział należy ocenić ilościowo przed zamrożeniem.

Wykonać porównanie osobno trenowanych modeli dla okien 0–190 ms, 0–390 ms i 0–790 ms. Wczesne okno też nie gwarantuje izolacji od opóźnionej odpowiedzi na poprzedni bodziec. Kontrolę przedbodźcową interpretować z uwzględnieniem RSVP i preprocessingu. Celem jest ustalenie, kiedy pojawia się informacja o bieżącej geometrii, a nie wybranie najkrótszego okna za wszelką cenę.

## Etap 2. Jeden badany i pomiar sygnału przed generacją

Przygotować `subject_id` w danych i modelach; wejście gałęzi przestrzennych ma jedną osobę. Zamienić wieloosobowy bank warunków na bank indeksowany obrazem, osobą, liczbą powtórzeń, splitem i hashem checkpointu. Zachować prawdziwy identyfikator badanego w metadanych, nawet jeśli lokalna oś tensora ma indeks 0.

Na pilotażu porównać encoder EEG do DINO i CLIP oraz mały model liniowy/ridge. Następnie wytrenować warianty krawędzi i koloru dla tej samej osoby. Na początku użyć istniejących zgrubnych celów i sprawdzonej architektury, bez dodawania kolejnych gałęzi, np. głębi.

Realny punkt odniesienia ze starego raportu hierarchicznej semantyki: top-1 dla pojedynczych osób wynosi od 7,68% do 23,52%, średnio 14,78%; dla fuzji dziesięciu osób 74,85%. Dotyczy to wewnętrznego testu 1654 obrazów, wspólnie uczonej przestrzeni i istniejącego modelu, nie nowego protokołu. Ten odczyt raportu nie jest nowym treningiem. Pokazuje, dlaczego nie należy przenosić wysokiego wyniku zespołu na projekt dla jednej osoby ani wybierać badanego po najwyższym wyniku testowym. [Źródło lokalne](D:/GSN_EEG/runs/eeg_hierarchical_encoder_eval/test_report.json).

Oceniać osobno kategorie/retrieval, geometrię i kolor. Kluczowy tani kontrolny model: przewidywanie geometrii wyłącznie z reprezentacji semantycznej S. Porównać go z przewidywaniem geometrii z S oraz dodatkowej gałęzi EEG E, z kontrolą pojemności. To ocenia korzyść E przed kosztownym treningiem generatora.

**Przejście dalej:** na walidacji gałąź geometrii wykazuje stabilny zysk ponad S na miarach geometrii. Jeżeli go brak, najpierw poprawiamy odczyt lub zmniejszamy rozdzielczość celu; większy generator nie rozstrzygnie tego problemu. Gałąź koloru jest opcjonalna, jeśli nie wykazuje dodatkowej korzyści.

## Etap 3. Sprawdzenie, czy generator potrafi wykorzystać warunek

Przed trenowaniem na EEG porównać: brak warunku, prawdziwy globalny embedding, prawdziwy embedding + prawdziwa zgrubna mapa struktury. Ten ostatni test jest kontrolą oracle: sprawdza generator, nie EEG.

PixArt kolegi pozostaje pierwszym punktem odniesienia. Dla struktur potrzebuje jawnej ścieżki przestrzennej, której obecny globalny adapter nie zapewnia. Jako oddzielny sprawdzian wykorzystać gotowy generator z kompatybilnym warunkowaniem obrazu i struktury, np. konfigurację SDXL z IP-Adapter i ControlNet. Wersje checkpointów i interfejsy sprawdzić przed implementacją; nie zakładać przenośności wag ControlNet SDXL do PixArt.

Taki podział semantyki i struktury ma precedens w [SGDM](https://arxiv.org/html/2604.22649v1), ale publikacja jest preprintem i jej wyników nie traktujemy jako gwarancji. Oddzielny generator referencyjny pozwala zbadać użyteczność map EEG niezależnie od dojrzałości nowego adaptera PixArt.

**Przejście dalej:** prawdziwa struktura poprawia geometrię generowanych obrazów. Jeśli nie, poprawiamy interfejs przestrzenny lub wybieramy skuteczniejszy generator na walidacji. Nie trenujemy długiego modelu EEG do nieskutecznego interfejsu. CLIP kontra DINO porównujemy zawsze przy tym samym generatorze.

## Etap 4. Minimalna wspólna architektura

### Co rzeczywiście oznaczają obecne cele krawędzi

`shape16` opisuje zgrubne zajęcie obszaru przez obiekt, `distance16` odległość od jego granic, `semantic32` granicę obiektu, a `crisp64` łączy kontur, krawędzie percepcyjne i zmiany głębi. Cele pochodzą z nauczycieli obrazowych, więc mogą dziedziczyć ich pomyłki; pomiar geometrii testowej powinien zawierać niezależnie sprawdzony podzbiór.

**`orientation64` nie oznacza kierunku ustawienia obiektu.** Jest polem normalnych obliczonych z gradientu mapy odległości. Funkcja kosztu używa wartości bezwzględnej iloczynu skalarnego, więc nie odróżnia znaku lokalnej normalnej. To zasadne dla niedirectionalnego kierunku granicy, ale metryka `orientation_cosine` nie dowodzi rozróżnienia nosa samolotu od ogona. Potrzebne są oddzielne adnotacje i pomiary kierunku obiektu oraz układu części.

**`confidence64` i `uncertainty64` nie są skalibrowaną pewnością EEG.** Ich cele wynikają z poparcia i niezgodności trzech nauczycieli widzących obraz. Model EEG uczy się odtwarzać tę własność targetu, a nie bezpośrednio przewidywać własny błąd. Co więcej, obecny bank warunków przekazuje `grid16`, a nie jawne mapy pewności do sterowania generacją.

Dlatego początkowo użyć stałej siły struktury wybranej na walidacji. Dopiero później porównać ją z siłą zależną od EEG: zgodności predykcji pomiędzy niezależnymi powtórzeniami, kilku modeli lub uczonego predyktora błędu. Każda taka pewność wymaga kalibracji poza treningiem. Zgodność powtórzeń nie gwarantuje poprawności: model może konsekwentnie odtwarzać tę samą błędną sylwetkę.

```text
EEG jednej osoby
  ├─ encoder semantyczny → S → dopasowanie → warunek semantyczny
  ├─ encoder struktury   → E → mapa/tokeny przestrzenne + pozycja
  └─ encoder koloru     → C → zgrubna paleta i rozmieszczenie barw
                                       ↓
                       zamrożony, sprawdzony generator
                       + uczone moduły warunkowania
                                       ↓
                               rekonstrukcja
```

Zacząć od S+E; C dołączyć w ablacjach. S oznacza wybraną na walidacji reprezentację, nie z góry zwycięskie DINO. Najpierw porównać CLIP i DINO globalne. SynCLR/fuzja to rozszerzenie po wybraniu działającego interfejsu.

Pierwszy wariant przestrzenny: miękka struktura w niskiej rozdzielczości lub tokeny z siatki 8×8, z jawnie zachowanym położeniem. Dla koloru zacząć od zwartego warunku, np. 4×4, i dopiero pomiarem uzasadnić większy. Istniejące wyjścia 16×16 można wykorzystać do takiej redukcji. Liczby te są propozycjami pilotażu, nie ustalonym optimum.

Utrzymać osobne projekcje i skalowanie S/E/C. W obecnym FusionPrior istnieją już znaczniki modalności i pozycje zapytań; przy ewentualnym wykorzystaniu tego modułu sprawdzić również zachowanie pozycji wejściowych tokenów przestrzennych. Nie traktować tokenów rozwiniętych z globalnego S jako lokalnych pomiarów EEG.

Adapter najpierw uczyć na dozwolonych obrazach treningowych, następnie dostosować do faktycznych predykcji EEG. Zacząć od prostego dopasowania, a prior probabilistyczny dodać tylko przy wykazanej korzyści. Predykcje do treningu mostu/adaptera zbierać poza foldem uczącym EEG encoder; dotyczy to wszystkich uczonych modułów dopasowania. Walidacji i testu nie używać do uczenia tego mostu.

Moduł generatywny musi mieć dostęp tylko do EEG i wyuczonych parametrów w inferencji. Prawdziwe etykiety, opisy, obrazy, krawędzie czy latenty testowego bodźca są dozwolone wyłącznie w oznaczonych kontrolach oracle.

Za precedens wielogałęziowości i prostszego dopasowania przyjąć [CognitionCapturerPro](https://arxiv.org/html/2603.12722v1). Za punkt odniesienia stabilnej przestrzeni przeznaczonej do generacji — [BrainHiVE](https://arxiv.org/html/2602.07495v1). To inspiracje i modele porównawcze, nie potwierdzenie skuteczności własnego połączenia.

Jeśli do eksperymentu wróci własny FusionPrior, przed uczeniem poprawić lub przemianować `gradient_loss` w [train_fusion_prior.py:108](D:/GSN_EEG/eeg_mae/training/train_fusion_prior.py:108). Obecna implementacja porównuje obcięte fragmenty latentów, a nie różnice pomiędzy sąsiednimi pozycjami. Jest dodatkową miarą zgodności wartości, nie bezpośrednim nadzorem gradientów. Nawet poprawny gradient latentu VAE nie jest automatycznie konturem obiektu w RGB; skuteczność geometryczną trzeba mierzyć po dekodowaniu.

## Etap 5. Eksperyment rozstrzygający

Minimalny zestaw osobno wytrenowanych wariantów przy jednym wybranym generatorze:

| Wariant | Funkcja |
|---|---|
| S | Punkt odniesienia |
| S+E | Zysk geometrii |
| S+C | Zysk koloru |
| S+E+C | Komplementarność |
| S z porównywalną dodatkową pojemnością | Kontrola liczby parametrów |
| S + geometria przewidziana wyłącznie z S | Kontrola geometrii możliwej do odgadnięcia z semantyki |

Dodatkowo w modelu wytrenowanym z opuszczaniem modalności: podmiana E pomiędzy dwoma bodźcami tej samej kategorii. To kontrola zależności od egzemplarza; sama podmiana może tworzyć nietypowe kombinacje, więc interpretować ją obok retrenowanych ablacji, nie samodzielnie.

Główna proponowana metryka: dwuwariantowe dopasowanie struktury rekonstrukcji do właściwego zdjęcia spośród dwóch zdjęć tej samej kategorii. Wynik losowy 50%; ocena oparta na ustalonej reprezentacji geometrii i pomocniczej ocenie ludzi. Do tego oddzielnie: błąd położenia, osi/orientacji, obrysu oraz relacji części na oznaczonym podzbiorze. Ocena części nie może sprowadzać się wyłącznie do osi głównej.

Kryterium potwierdzenia hipotezy: S+E poprawia główną metrykę względem S i kontroli geometrii z S, z dodatnią dolną granicą sparowanego 95% przedziału ufności oraz bez materialnej utraty semantyki. Proponowany praktyczny próg pilotażu: zysk co najmniej 5 punktów procentowych w dopasowaniu struktury, przy spadku rozpoznawalności kategorii nie większym niż 2 punkty procentowe. Progi ustalić przed oceną testu; nie są obietnicą wyniku.

Na walidacji selekcjonować modele; test otworzyć po zamrożeniu konfiguracji. Bootstrap grupować według kategorii; powtórzenia EEG i seedy generatora nie stanowią niezależnych osób. Pilotaż jeden seed, potwierdzenie trzy seedy modelu i te same z góry ustalone seedy generacji we wszystkich wariantach. Nie wybierać najlepszego wygenerowanego obrazka według prawdziwego celu.

Wynik sukcesu: czytelny obiekt o lepszej, bodźcowo specyficznej geometrii, potwierdzony metrykami. Wynik negatywny również jest rozstrzygający: np. kolor pomaga, krawędzie nie; albo użyteczny sygnał obejmuje orientację, lecz nie konfigurację części. Dalsze twierdzenia ograniczyć do wykazanego poziomu.

## Sprzęt, kolejność i koszt

Lokalnie dostępna jest RTX 3060 Laptop 6 GB. To podstawa do planowania testów, cache i pilotażu małych encoderów, ale nie potwierdzenie, że każdy trening PixArt/SDXL zmieści się w pamięci. Dostęp do GPU kolegi lub Lightning.ai pozostaje do ustalenia. Nie uruchomiono płatnej instancji ani treningu.

Przed szacowaniem godzin wykonać pomiar małej liczby kroków i pełnej walidacyjnej ścieżki generacji: czas/krok, peak VRAM, pamięć RAM, wielkość cache, koszt checkpointów. Następnie policzyć budżet całego etapu. Banki nauczycieli i latenty VAE przygotowywać raz; encodery EEG, VAE i generator nie muszą jednocześnie zajmować GPU przy pracy na cache.

Nie wykonywać pełnego iloczynu reprezentacje × generatory × osoby × seedy × modalności. Kolejność ogranicza koszt: wspólna baza → dane jednej osoby → pomiar geometrii → oracle generatora → wybór semantyki → S/E/C → potwierdzenie na kolejnych osobach. Silniejsze GPU umożliwi większy eksperyment, lecz nie zmienia wymogu niezależnego testu.

Pierwszy konkretny pakiet implementacyjny obejmuje wyłącznie etap 0 i 1 oraz obsługę jednej osoby z etapu 2. Jego rezultatem powinny być: branch z zachowaną pracą obu autorów, poprawny Lightning, manifest 6/2/2, jednoznaczne warunki inferencji i krótki przebieg sprawdzający. Dopiero po nim uruchamiany jest pilotaż uczenia.

## Powiązane materiały

- [Analiza architektury, wyników i literatury](ANALIZA_EEG_STRUKTURA_DINO_CLIP.md).
- [Przegląd commitów SSSHANKS](REVIEW_SSSHANKS.md).
- [Encoder krawędzi](D:/GSN_EEG/eeg_mae/models/edge_specialist.py), [encoder koloru](D:/GSN_EEG/eeg_mae/models/color_specialist.py).
- [Obecny bank warunków](D:/GSN_EEG/eeg_mae/data/build_fusion_condition_bank.py).
- [Adapter PixArt kolegi](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/lightning/dit_adapter.py).
