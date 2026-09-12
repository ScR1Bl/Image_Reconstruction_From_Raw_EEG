# EEG → obraz: semantyka, rzeczywista struktura i wybór reprezentacji

Analiza kodu, zapisanych wyników i literatury. Stan: 12 września 2026.

## 1. Wnioski decyzyjne

1. **Warto zachować krawędzie i kolor jako kandydatów do eksperymentu. Ich konieczność nie jest jeszcze wykazana.** Obie niezależne gałęzie EEG mają wyniki lepsze od przemieszanych par, ale brakuje testu, czy poprawiają rekonstrukcję ponad samą semantykę.
2. **DINO jest uzasadnioną alternatywą dla CLIP, a nie potwierdzonym rozwiązaniem problemu.** Przewaga 0,1038 nad 0,0887 to około 17% względnej zmiany nieustalonej metryki. Nie oznacza 17% więcej informacji ani 17% lepszych obrazów.
3. **Najpilniejszy problem to ustalenie miejsca utraty informacji.** Skrypt oceny adaptera SSSHANKS używa cech prawdziwych obrazów. Dopóki nie ma osobnego wyniku z przewidywanymi cechami EEG, nie można przypisać jakości pokazanych rekonstrukcji dekodowaniu EEG.
4. **Osiem tokenów utworzonych z jednego globalnego embeddingu nie jest ośmioma odczytanymi fragmentami obrazu.** Adapter może nauczyć się korzystać z tych tokenów, ale nie odzyska pominiętej w wejściu informacji o konkretnym bodźcu.
5. **Sam pomysł semantyka + krawędzie/kolor + generator ma wcześniejsze odpowiedniki.** Mocniejszą tezą badawczą jest wykazanie, że EEG dostarcza geometrii konkretnego egzemplarza ponad wiedzę o jego kategorii i ponad globalne embeddingi.
6. **Lightning warto rozpatrywać jako osobny refaktor odtwarzalności.** Nie ma podstaw, żeby oczekiwać poprawy klasyfikacji wyłącznie po zmianie frameworka.

W tej analizie nie uruchomiono treningu, nie zmieniono kodu projektu, nie utworzono nowej gałęzi ani commitu. Powstał wyłącznie ten raport poza katalogiem roboczym projektu. Ocena jakości opiera się na istniejących raportach i panelach; wyniki treningów nie zostały niezależnie odtworzone.

## 2. Jakie wersje kodu zostały rozdzielone

Lokalny `main` wskazuje `677ee8d`. SSSHANKS jest autorem, a jego gałąź to `origin/s_dev`, ze szczytem `89142c2`. Widoczne commity autora: `b9360cc`, `d36519c`, `23d83f8`, `89142c2`. Snapshot gałęzi przeanalizowano w osobnym worktree. Szczegółowe problemy wykonania i checkpointów są opisane w [przeglądzie commitów](REVIEW_SSSHANKS.md).

**Gałąź SSSHANKS:** adaptacja istniejących modeli do Lightning, fuzja pomiędzy osobami, nowe cele semantyczne i głowica CLIP, następnie adapter zamrożonego PixArt. Kolor i krawędzie nie warunkują tego adaptera. Fuzja osób poprawia estymację odpowiedzi na bodziec, lecz nie jest fuzją koloru, geometrii i semantyki.

**Obecny katalog `D:/GSN_EEG`:** zawiera niezacommitowane, niezależne gałęzie koloru i krawędzi, hybrydowy kolor, hierarchiczną semantykę oraz własny `FusionPrior`. To inny stan niż commit kolegi; nie można przypisywać wszystkich obecnych wyników jednej architekturze.

Obecny `FusionPrior` jest własnym modelem rectified flow do latentów SDXL VAE, domyślnie 4×16×16, odpowiadających obrazowi 128×128. Adapter kolegi wykorzystuje zamrożony PixArt 512×512. Zmiana między nimi zmienia zarówno model generatywny, jak i rozdzielczość oraz procedurę uczenia. Nie byłaby czystą ablacją CLIP kontra DINO.

## 3. Co pokazuje przesłany panel

Panel zawiera osiem oryginałów i osiem wyników. Są to kategorie od `aardvark` do `airbag`, w kolejności alfabetycznej. To pasuje do sposobu wyboru początkowych przykładów w kodzie oceny, ale **nie potwierdza pochodzenia panelu ani rodzaju wejścia do generatora**.

| Bodziec | Co zachowano w wyniku | Co jest osłabione lub zmienione |
|---|---|---|
| Mrównik | Brązowy ssak, podłoże, zgrubna centralna sylwetka | Zwierzę przypomina małego gryzonia; pysk, uszy i proporcje nie odpowiadają oryginałowi |
| Liczydło | Powtarzalne brązowe elementy | Znika czytelna konstrukcja ramy i prętów; dominują walcowate bryły |
| Akordeon | Czerwień, jasne punkty, częściowo charakterystyczne elementy | Korpus i rozmieszczenie części są zdeformowane |
| Żołędzie | Brązowe drobne obiekty na podłożu | Kształty przechodzą w kamienie i stożkowate bryły, tożsamość jest niepewna |
| Klimatyzator | Jasna ściana i prostokątny obiekt | Brak czytelnych żaluzji i detali urządzenia |
| Materac | Duża niebieska powierzchnia | Zanika podział na komory, proporcje i funkcja przedmiotu |
| Pompka | Czerwone elementy i fragment otoczenia | Układ rur, podstawy i uchwytu nie tworzy wyraźnie tej samej konstrukcji |
| Poduszka powietrzna | Jasna poduszka we wnętrzu samochodu | Układ wnętrza i lokalna geometria są inne |

**Interpretacja:** na tych przykładach zachowanie palety i lokalnych powtarzalnych faktur jest mocniejsze niż zachowanie konfiguracji części i tożsamości obiektu. Nie jest to jeszcze mocny dowód odczytania „prawdziwej struktury”. Nawet prawidłowa ogólna sylwetka może powstać z prioru kategorii: generator wie, jak zwykle wygląda zwierzę lub instrument.

To osiem wybranych przykładów, więc nie pozwalają oszacować częstości błędów na całym zbiorze. Panel sam w sobie nie rozstrzyga też, czy pomyłka pojawiła się w EEG encoderze, czy w adapterze.

Kluczowy dowód z kodu: [eval_dit_adapter.py:90](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/evaluation/eval_dit_adapter.py:90) ustawia `cond = clip_bank["clip_global"][rows]`, czyli cechy prawdziwych obrazów. W tej ścieżce nie ma inferencji EEG. Jeśli panel pochodzi z tego przebiegu, obserwujemy już błędy rekonstrukcji z prawidłowego warunku obrazowego. Poprawa EEG nie usuwa automatycznie tego ograniczenia generatora.

## 4. Czy kolor i krawędzie powtarzają tę samą informację?

### Różnica między obrazem, reprezentacją i jej predykcją

Z pełnego RGB można wyznaczyć krawędzie. W tym sensie mapa krawędzi nie dodaje informacji do idealnego, niekompresowanego obrazu. Ale wejściem generatora nie jest idealne RGB: są nim skompresowane, obarczone błędem predykcje z EEG.

Mapa koloru może zachowywać paletę i zgrubne rozmieszczenie barw, tracąc cienkie granice. Krawędzie opisują lokalizację granic, orientację i konfigurację części, ale bez koloru nie rozróżnią np. identycznych czerwonych i zielonych przedmiotów. Semantyczny embedding może rozpoznawać „krzesło”, pomijając stronę, w którą jest skierowane, lub dokładny układ nóg.

Podobieństwo wizualne wyjść gałęzi może pochodzić ze wspólnej niskoczęstotliwościowej sylwetki. Nie dowodzi redundancji ich ukrytych tokenów. Również odmienne wyglądające mapy nie dowodzą komplementarności — mogą kodować tę samą kategorię innym stylem.

Wszystkie gałęzie są funkcjami tego samego EEG. Nie wytwarzają nowej informacji o bodźcu; ich korzyść polega na innym nadzorze, ograniczeniach i zachowaniu różnych aspektów sygnału dostępnych dla modeli o skończonej pojemności.

### Co już potwierdzają lokalne liczby

| Gałąź i metryka | Poprawne pary | Przemieszane pary | Odczyt |
|---|---:|---:|---|
| Kolor: RGB MAE ↓ | 0,1701 | 0,2566 | Wyraźny sygnał ogólnego wyglądu |
| Kolor: chroma MAE ↓ | 0,0400 | 0,0625 | Sygnał barwy poza samą luminancją |
| Kolor: korelacja palety ↑ | 0,5305 | 0,1383 | Rozróżnianie palet pomiędzy bodźcami |
| Kolor: chroma gradient MAE ↓ | 0,013161 | 0,013217 | Bardzo mała różnica dla tej miary lokalnych przejść |
| Krawędzie: shape IoU ↑ | 0,5641 | 0,4232 | Zgrubny kształt lepszy niż kontrola |
| Krawędzie: semantic tolerant F1 ↑ | 0,6193 | 0,4737 | Sygnał granic przy tolerancji lokalizacji |
| Krawędzie: crisp tolerant F1 ↑ | 0,3823 | 0,3058 | Szczegółowe krawędzie znacznie słabsze |

Źródła: [kolor, 1654 obrazy](D:/GSN_EEG/runs/eeg_color_encoder_direct/test_report.json), [krawędzie, 1354 obrazy](D:/GSN_EEG/runs/eeg_edge_encoder_direct/test_report.json). Zbiory mają różną liczebność; filtr poprawności celów krawędzi zmniejsza próbę. Nie jest to bezpośredni ranking gałęzi. Kontrola z przemieszaniem usuwa również zgodność kategorii, więc nie izoluje informacji geometrycznej ponad kategorię.

Obejrzane panele walidacyjne niezależnych gałęzi pokazują głównie rozmyte pola barw oraz szerokie obrysy; szczegółowe krawędzie zawierają także regularne artefakty. To przemawia za sprawdzeniem miękkiego, zgrubnego warunkowania przestrzennego zamiast wymuszania ostrych konturów. Nie należy przenosić oceny kilku paneli na cały test.

Wariant [hybrid_safe](D:/GSN_EEG/runs/eeg_color_encoder_hybrid_safe/test_report.json) ma status `qualified: true`, lecz np. salient chroma recall wynosi około 0,433 wobec 0,420 po przemieszaniu. Nazwa „qualified” oznacza przejście lokalnych kryteriów, a nie dowód znaczącej poprawy końcowego generatora.

**Decyzja:** krawędzie zachowałbym jako hipotezę dotyczącą geometrii, a kolor jako osobną, możliwie zwartą ścieżkę palety i rozmieszczenia chromatycznego. Liczbę tokenów i potrzebę osobnych dużych encoderów powinny rozstrzygnąć ablacje. Warto porównać też wspólny encoder EEG z odrębnymi głowicami: jest tańszy, ale może powodować konflikt celów i utratę szczegółów w wspólnym wąskim gardle.

## 5. CLIP, DINO i argument o 17%

CLIP uczy się relacji obraz–tekst; dlatego ma naturalne zastosowanie do nazw kategorii i opisów. DINOv2 uczy cech wizualnych bez par tekstowych i oferuje cechy globalne oraz lokalne. Nie wynika z tego podział „CLIP tylko semantyka, DINO tylko geometria”: oba reprezentują oba aspekty w różnym stopniu. [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html), [DINOv2](https://arxiv.org/abs/2304.07193).

W badaniach EEG wybór nauczyciela ma duże znaczenie, lecz nie ma uniwersalnego zwycięstwa każdej wersji DINO nad każdą wersją CLIP. Porównanie ICML 2025 obejmuje różne rodziny modeli oraz reprezentacje dostrajane do ocen podobieństwa dokonywanych przez ludzi; SynCLR i takie dostrojenie są ważnymi punktami odniesienia. [Rajabi i in., 2025](https://proceedings.mlr.press/v267/rajabi25a.html).

### Co w propozycji kolegi jest trafne

Własny adapter może nauczyć się przyjmować DINO. Nazwa argumentu `clip_dim` nie stanowi ograniczenia matematycznego. Istnieje także dowód poza EEG, że DINO może warunkować dyfuzję: Semantica bada DINOv2 i SigLIP oraz lokalne cechy podawane przez cross-attention. To uzasadnia wykonalność, a nie oczekiwany wynik tego konkretnego projektu. [Semantica](https://arxiv.org/html/2405.14857v3).

Jeżeli trunk A ma lepszą generalizację na tym samym protokole, warto go zachować jako punkt odniesienia i sprawdzić adapter do jego reprezentacji. Nie ma potrzeby uznawania finalnego CLIP za jedyny dopuszczalny cel EEG.

### Co wymaga korekty

**K i V nie startują w tym kodzie całkowicie od zera.** [dit_adapter.py:143](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/lightning/dit_adapter.py:143) kopiuje wagi i biasy `attn2.to_k` oraz `attn2.to_v` do ścieżki obrazu. Są następnie uczone; projekcja obrazu jest nowa, skala gałęzi startuje od zera, a Q i projekcja wyjścia korzystają z zamrożonego transformera. To nie blokuje DINO, ale obala dokładne uzasadnienie „wszystko i tak od zera”. Natywna ścieżka tekstowa tego PixArt używa T5; CLIP jest tu dodatkowym źródłem warunku obrazowego.

Arytmetyka:

`(0,1038 − 0,0887) / 0,0887 = 17,02%`.

Różnica bezwzględna to 0,0151. Gdyby chodziło o top-1, byłoby to 1,51 punktu procentowego; gdyby o MRR lub korelację, interpretacja jest inna. Nie znaleziono lokalnego raportu identyfikującego te wartości i „bieg A”. Przyjmujemy je jako przekazaną informację, nie zweryfikowany wynik. Szczególnie **korelacji lub cosinusa z dwóch różnych przestrzeni nie można traktować jako skalibrowanej miary ilości informacji**.

Nawet przy tej samej metryce retrieval potrzebne są: identyczne obrazy, pula kandydatów, osoby, liczba powtórzeń EEG, wybór checkpointu na walidacji oraz ocena zmienności między seedami. Porównanie zmieniające jednocześnie nauczyciela, trunk i schemat treningu nie izoluje kosztu przestrzeni docelowej.

DINO musi przejść dwa sprawdziany: **czy EEG przewiduje go lepiej oraz czy z takich przewidywanych cech generator tworzy wierniejszy bodziec**. Pierwszy nie gwarantuje drugiego.

### Cztery zasadne warianty

| Wariant | Co rozstrzyga |
|---|---|
| EEG → CLIP → adapter PixArt | Obecny punkt odniesienia |
| EEG → DINO → nowo wytrenowany adapter PixArt | Bezpośredni test propozycji kolegi |
| EEG → DINO → mały most do przestrzeni generatora | Oddzielenie łatwego celu EEG od stabilnego interfejsu generatora |
| EEG → DINO + CLIP → adapter | Czy reprezentacje wnoszą użyteczną komplementarność |

Najpierw należy porównać globalne embeddingi przy tym samym generatorze i budżecie. Osobny eksperyment może dodać rzeczywiste przewidywane cechy przestrzenne, np. 4×4 lub 8×8, z jawną informacją o położeniu. [ImageProjection:92](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/lightning/dit_adapter.py:92) jedynie rozwija globalny wektor do ośmiu tokenów. Zwiększenie ich liczby nie jest samo w sobie zwiększeniem liczby odczytanych szczegółów.

## 6. Skąd mogą brać się nieczytelne obiekty i jak poprawić semantykę

### Najpierw diagnostyka przy zamrożonym generatorze

| Warunek wejściowy | Rola eksperymentu |
|---|---|
| Brak warunku / stały warunek | Ile wyglądu zapewnia sam prior generatora |
| Cechy prawdziwego obrazu, czyli oracle | Czy reprezentacja i adapter wystarczają do odtworzenia obiektu |
| Cechy przewidziane z EEG na niewidzianych obrazach | Rzeczywista jakość całego systemu |
| Te same predykcje po dopasowaniu do przestrzeni generatora | Czy głównym problemem jest rozbieżność rozkładów |
| Prawdziwa etykieta kategorii jako dodatkowy warunek, tylko diagnostycznie | Ile daje idealna wiedza o kategorii, a ile szczegółów nadal brakuje |

Warianty powinny korzystać z tych samych bodźców i seedów generatora. Oracle jest kontrolą diagnostyczną; nie wolno mieszać jego wyników z wynikami inferencji wyłącznie z EEG.

**Jeżeli oracle jest już słaby:** sprawdzić moc warunku i wartości uczonych skal adaptera, liczbę kroków, CFG wybierane na walidacji, pojemność projekcji, zakres danych uczących adapter oraz przewagę lokalnych cech nad wektorem globalnym. Zbyt słaby warunek może być ignorowany; zbyt duża skala może nasilać błędy. Żaden kierunek zmiany CFG nie jest z góry poprawny.

**Jeżeli oracle jest dobry, a EEG słabe:** zbadać predykcję semantyki oraz dopasowanie cech. Kontrastywna zgodność kierunku nie gwarantuje zgodności norm, wariancji, kowariancji i rozkładu błędów. Adapter uczony na czystych cechach obrazów może źle działać na pozornie podobnych cechach EEG.

W skrypcie kolegi symulowany szum ma skalę z `cond.std()`. To nie jest zmierzony rozkład błędów EEG. Co więcej, metryka porównuje wygenerowany obraz do podanego, także zaszumionego warunku. Do oceny rekonstrukcji referencja powinna pozostać prawdziwym bodźcem. Losowy izotropowy szum może być pomocną analizą odporności, ale nie zastępuje testu na rzeczywistych predykcjach.

Najtańszy most do sprawdzenia to regularyzowana regresja albo mały MLP. Dopiero jego ograniczenia uzasadniają bardziej złożony prior probabilistyczny. Trening odporności adaptera warto oprzeć na prawdziwych błędach EEG. Predykcje do takiego treningu powinny być uzyskane poza foldem uczącym encoder; predykcje dla obrazów zapamiętanych przez encoder mogą być nierealistycznie dobre. Dyfuzyjne dopasowanie reprezentacji EEG do generatora ma precedens w ATM; z kolei CogCapPro rozważa prostsze dopasowanie MLP. [ATM, NeurIPS 2024](https://arxiv.org/html/2403.07721v4), [CognitionCapturerPro, preprint 2026](https://arxiv.org/html/2603.12722v1).

### Semantyka, która nie usuwa geometrii

Jeżeli model rzeczywiście myli kategorie już przed generowaniem, uzasadnione są: pomocniczy cel obraz–tekst, miękka klasyfikacja hierarchiczna oraz zachowanie równoległego celu rozróżniającego konkretne obrazy. Samo mocne ściąganie wszystkich przykładów jednej kategorii do wspólnego punktu może poprawić nazwę obiektu kosztem orientacji i układu części.

Przy wielu osobach oglądających ten sam obraz warto traktować ich odpowiedzi jako wiele zgodnych przykładów bodźca, a nie błędne negatywy. Zgodność kategorii jest słabszą relacją niż zgodność konkretnego zdjęcia; funkcja kosztu powinna je rozróżniać.

Na nowych kategoriach właściwym punktem odniesienia jest predykcja otwartosłownikowa, np. względem tekstowych prototypów kandydatów. Głowica trenowana na etykietach końcowego testu nie byłaby testem zero-shot. Prawdziwe etykiety lub opisy testowego bodźca nie powinny być podawane w podstawowej ścieżce EEG → obraz.

Jeśli rozkład przewidywanych kategorii jest niepewny, twarde wpisanie jednej nazwy może stworzyć wyraźny, lecz błędny obiekt. Alternatywą do oceny są osobne próbki dla kilku hipotez z podaną pewnością. Mieszanie reprezentacji niezgodnych hipotez może prowadzić do hybryd; jest to możliwe wyjaśnienie części artefaktów, a nie potwierdzona diagnoza tego panelu.

Gałęzie przestrzenne powinny wpływać proporcjonalnie do swojej zwalidowanej wiarygodności. Duża pewność koloru nie oznacza dużej pewności drobnego konturu. Dla obecnych map zgrubny obrys, mapa odległości i orientacji są bardziej uzasadnionymi kandydatami niż wymuszanie ostrych krawędzi wszędzie. Wyniki UBP wskazują na korzyść dostosowania wizualnych celów do ograniczonej szczegółowości sygnału EEG; nie dowodzą, że w EEG nigdy nie ma informacji o detalach. [UBP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Bridging_the_Vision-Brain_Gap_with_an_Uncertainty-Aware_Blur_Prior_CVPR_2025_paper.pdf).

## 7. Co oznacza obecny wysoki wynik semantyki hierarchicznej

[Lokalny raport](D:/GSN_EEG/runs/eeg_hierarchical_encoder_eval/test_report.json) podaje `fused top1 = 0,74849` i `MRR = 0,83153`. To wynik zespołu 10 osób, na wewnętrznym teście 1654 obrazów, w uczonej przestrzeni fuzji. Nie oznacza 74,85% poprawnie wygenerowanych obrazów, wyniku jednej osoby ani 200-way zero-shot na oficjalnym teście. W kontrolach zerowego EEG i przemieszania wyniki są bliskie szansy, co jest potrzebnym pozytywnym sygnałem, lecz nie testem całej ścieżki generacji.

Wewnętrzny podział wykorzystuje inne obrazy tych samych 1654 kategorii: osiem na trening, jeden na walidację, jeden na test. Oficjalne 200 kategorii jest odrębnym sprawdzianem generalizacji. Trzeba również oddzielić uśrednianie powtórzeń jednej osoby od fuzji odpowiedzi różnych osób. Żadnego z tych wyników nie należy porównywać bezpośrednio z publikacjami stosującymi inne pule kandydatów i liczbę powtórzeń.

Istotny szczegół architektury: [hierarchical_semantic.py:113](D:/GSN_EEG/eeg_mae/models/hierarchical_semantic.py:113) tworzy wszystkie głowice z tego samego wyjścia EEG o wymiarze 384. Dodatkowe głowice dostarczają dodatkowego nadzoru, ale nie są niezależnymi odczytami surowego sygnału.

Ponadto [train_hierarchical_semantic_encoder.py:186](D:/GSN_EEG/eeg_mae/training/train_hierarchical_semantic_encoder.py:186) uczy jednocześnie encoder EEG i projekcje fuzji po stronie obrazów. Cel może więc dostosować się do tego, co łatwo przewidywać z EEG, bez wymogu zachowania wszystkich informacji potrzebnych generatorowi. To poprawna możliwość uczenia reprezentacji, lecz wynik retrieval nie potwierdza gotowego interfejsu generatywnego. Głowica nazwana `clip_b32` działa tu w przestrzeni projekcji; nie jest automatycznie natywnym embeddingiem CLIP.

BrainHiVE w wariancie rekonstrukcji najpierw uczy fuzję wraz z interfejsem generatora na danych obrazowych, następnie zamraża tę przestrzeń i dopasowuje do niej mózg. Lokalna wspólnie uczona fuzja nie odtwarza dokładnie tego porządku. To ważniejsze dla generowania niż podobieństwo nazw modułów. [BrainHiVE, sekcja 3.4](https://arxiv.org/html/2602.07495v1).

## 8. Najbliższe wcześniejsze rozwiązania

Stan publikacji i preprintów poniżej dotyczy dostępnych źródeł na dzień analizy. Wyniki autorów nie zostały niezależnie odtworzone. Tabela porównuje idee; nie tworzy rankingu metryk z niezgodnych protokołów.

| Praca | Znaczenie dla tej architektury | Różnica lub ograniczenie porównania |
|---|---|---|
| [Joint Learning… Hierarchical Representation…, UniReps 2024](https://proceedings.mlr.press/v285/akbari24a.html) | Hierarchiczne ujęcie różnych poziomów cech obrazu w rekonstrukcji EEG | Inne dane i generator; wcześniejszy punkt odniesienia dla wielopoziomowego nadzoru |
| [ATM, NeurIPS 2024](https://arxiv.org/html/2403.07721v4) | EEG, reprezentacja wizualna, prior dopasowania i dyfuzja | Pokazuje potrzebę analizy przejścia między embeddingiem EEG a generatorem |
| [CognitionCapturer, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33587) | Osobni eksperci EEG dla obrazu, tekstu i głębi, następnie dopasowanie i generator | Sama wielogałęziowość oraz eksperci modalności nie są nowością |
| [Human-Aligned Image Models…, ICML 2025](https://proceedings.mlr.press/v267/rajabi25a.html) | Porównanie nauczycieli wizualnych i dostrojenia do ludzkiego podobieństwa | Ważny argument za badaniem SynCLR i konkretnych reprezentacji, nie tylko etykiet „CLIP/DINO” |
| [UBP, CVPR 2025](https://arxiv.org/abs/2503.04207) | Niepewność i rozmycie celów wizualnych | Ostrość nauczyciela nie jest równoznaczna z użytecznością dla EEG |
| [BrainHiVE, ICLR 2026](https://arxiv.org/html/2602.07495v1) | Fuzja różnych wizualnych embeddingów oraz VAE | Bliski punkt odniesienia dla obecnej semantyki hierarchicznej |
| [CognitionCapturerPro, preprint 13.03.2026](https://arxiv.org/html/2603.12722v1) | Obraz, tekst, głębia, krawędzie; niepewność, fuzja i prostsze dopasowanie | Bardzo bliski precedens; samo dodanie krawędzi i wag niepewności nie wystarcza jako nowy wkład |
| [JMVR, preprint 20.03.2026](https://arxiv.org/html/2603.19667v1) | DiT, niezależne warunkowanie EEG i tekstem, cechy krawędzi, nasycenia oraz głębi | Metoda używa także opisów tekstowych; nie jest bezpośrednio tym samym zadaniem co inferencja tylko z EEG |
| [SGDM, preprint 24.04.2026](https://arxiv.org/html/2604.22649v1) | Jawna ścieżka struktury obok semantyki, IP-Adapter i ControlNet | Kolejny precedens osobnego sterowania semantyką i strukturą |
| [NVOL, preprint 02.09.2026](https://arxiv.org/html/2609.02582v1) | Reprezentacja warstwy pośredniej jako cel EEG, następnie most do CLIP i generatora | Bardzo świeży wynik; wspiera rozdzielenie celu EEG i interfejsu generatora, wymaga niezależnej oceny |

Nie ma podstaw do twierdzenia, że żadna praca nie wykorzystała podobnego zestawu składników. Również proponowane poniżej eksperymenty nie stanowią dowodu pierwszeństwa pomysłu. Przeszukanie literatury ujawnia istotne nakładanie się idei, ale nie zastępuje pełnej oceny nowości przygotowywanej publikacji.

## 9. Jak wykazać dodatkową, prawdziwą strukturę

Najmocniejsze pytanie brzmi: **czy gałąź przestrzenna pomaga odzyskać orientację, pozycję i konfigurację części konkretnego bodźca po uwzględnieniu tego, co już wiadomo z semantyki?**

Samo lepsze SSIM, kolor albo DINO-similarity może być skutkiem poprawniejszej kategorii. Przykładowo rozpoznanie samolotu pozwala priorowi wygenerować skrzydła, ale nie dowodzi odczytania ich orientacji z EEG.

Podstawowa macierz eksperymentów:

| Wariant | Pytanie |
|---|---|
| S: tylko semantyka | Punkt odniesienia |
| S + E | Czy krawędzie dają dodatkową geometrię? |
| S + C | Czy kolor daje dodatkową paletę i rozmieszczenie barw? |
| S + E + C | Czy obie ścieżki współpracują? |
| S + dodatkowe parametry, bez nowej ścieżki EEG | Czy zysk pochodzi tylko z większej pojemności? |
| S + mapa krawędzi przewidziana wyłącznie z S | Czy osobny encoder EEG wnosi coś ponad geometrię przewidywalną z semantyki? |
| S + E/C niedopasowane do egzemplarza, ale dopasowane kategorią | Czy korzyść dotyczy konkretnego bodźca, a nie etykiety? |

Potrzebne są zarówno osobno wytrenowane warianty, jak i wyłączanie gałęzi w tym samym modelu. Samo zerowanie tokenów może wprowadzać wejście spoza rozkładu treningowego; pomaga trenowanie z opuszczaniem modalności i porównywanie z kontrolami zachowującymi rozkład.

**Ograniczenie obecnego podziału:** jeden obraz na kategorię w wewnętrznym teście nie pozwala na mocny test rozróżniania kilku egzemplarzy tej samej kategorii. Oficjalny test również ma po jednym obrazie na kategorię. Potrzebny jest dodatkowy, zablokowany podział z co najmniej dwoma obrazami na kategorię albo odpowiednio zaplanowane predykcje poza foldem. Należy zachować dotychczasowy benchmark osobno i nie stroić nowego protokołu na wynikach testu.

Ocena powinna osobno raportować:

- **Tożsamość:** retrieval EEG i rozpoznawalność wygenerowanego obiektu; ocena obrazów przez niezależny model oraz osoby niewidzące etykiety celu.
- **Geometrię konkretnego bodźca:** orientację głównej osi, położenie, proporcje, kontur, relacje części i zgodność lokalnych cech; szczególnie przy porównaniach wewnątrz kategorii.
- **Kolor:** paletę oraz przestrzenne położenie chromy, nie tylko RGB MAE.
- **Realizm:** jako oddzielną własność. Fotorealistyczny niewłaściwy obiekt jest błędną rekonstrukcją.

Ocena przez model z tej samej rodziny, która była celem uczenia, jest użyteczna, ale nie powinna być jedynym sędzią. W skrypcie kolegi `concept_acc` wynika z najbliższego warunku w puli; nie jest niezależną klasyfikacją obrazu. Przy jednym przykładzie danej kategorii praktycznie sprowadza się do dopasowania w tej puli.

Wyniki wymagają sparowanych przedziałów niepewności, kilku seedów oraz jawnej liczby osób i powtórzeń EEG. Nie należy wybierać najlepszej próbki generatora na podstawie podobieństwa do prawdziwego obrazu testowego i przedstawiać jej jako przeciętnej skuteczności.

Obiecujący kierunek wkładu badawczego to **wykrywanie i wykorzystywanie tylko tej geometrii, dla której EEG daje wiarygodną informację dodatkową względem semantyki**, z udokumentowanym zakresem rozdzielczości i niepewnością. Oryginalność wymagałaby konkretnej metody oraz przewagi nad bliskimi modelami, zwłaszcza CogCapPro i SGDM; sama deklaracja „uncertainty-aware” już występuje w literaturze.

## 10. Kolejność dalszych działań — propozycja, bez wykonania

1. Zidentyfikować checkpoint i skrypt panelu; zapisać dokładną definicję 0,1038 i 0,0887 oraz konfiguracje obu biegów.
2. Ustalić poprawną ewaluację i wykonać rozdzielone próby oracle / EEG / EEG po dopasowaniu. To wskaże, czy pierwszeństwo ma adapter, czy encoder.
3. Porównać CLIP i DINO z odpowiadającymi im adapterami na tym samym PixArt. Nie zmieniać jednocześnie generatora, rozdzielczości i protokołu danych.
4. Sprawdzić prosty most reprezentacji i odporność na rzeczywiste błędy EEG. Zachować aktualny SynCLR/CLIP/VAE jako osobny punkt odniesienia, z poprawną interpretacją jego przestrzeni.
5. Na najlepszym zwalidowanym torze semantycznym wykonać S / S+E / S+C / S+E+C i kontrole informacji ponad kategorię.
6. Dopiero z tych wyników zdecydować o liczbie tokenów, oddzielnych encoderach, mechanizmie fuzji i treści tezy o nowości.

### Lightning i integracja

Docelowy refaktor może powstać np. na osobnej gałęzi `research/eeg-structure-lightning`, ale nie został wykonany. Najpierw warto zachować niezmienione obliczenia i porównać forward, loss oraz pojedynczy krok optymalizatora z obecną implementacją. Następnie sprawdzić EMA, flagi zamrożenia, kolejność loaderów, identyfikatory osób, precision i pełne wznowienie z optimizerem, schedulerem oraz stanami RNG.

Wcześniejszy przegląd wykazał konkretne przeszkody w bezpośrednim przeniesieniu gałęzi kolegi: niezgodność domyślnego wieloloaderowego przebiegu semantyki z Lightning, niezgodne importy po zmianach obecnych modeli, dodatkowe wagi VAE w checkpointach po podglądzie oraz wznowienie treningu adaptera ograniczone do wag. Są też problemy interpretacji metryk retrieval przy remisach. Część dotyczy odziedziczonych narzędzi, więc nie należy przypisywać wszystkich błędów autorowi nowych commitów. Reprodukcje i szczegóły są w [osobnym przeglądzie](REVIEW_SSSHANKS.md).

Przejście testów jednostkowych nie zastępuje sprawdzenia kompletnego treningu i odtwarzalności. Naprawa infrastruktury i eksperyment DINO powinny stanowić oddzielne zmiany, aby było wiadomo, skąd pochodzi różnica wyniku.

## 11. Granice ustaleń i materiały

**Potwierdzone w kodzie:** autor i gałąź, wejście obrazowe skryptu DiT, inicjalizacja K/V, globalne źródło ośmiu tokenów, wspólne uczenie fuzji obrazowej, różnice architektur i protokołów.

**Potwierdzone jako zawartość zapisanych raportów:** wartości metryk koloru, krawędzi i semantyki. Nie są to ponownie uruchomione eksperymenty.

**Obserwacja wizualna:** opis ośmiu rekonstrukcji z przesłanego panelu i lokalnych paneli gałęzi. Nie zastępuje ilościowej oceny całego zbioru.

**Nierozstrzygnięte:** czy przesłany panel powstał z rzeczywistego EEG, czym dokładnie są liczby obu biegów, czy DINO poprawi końcową rekonstrukcję, czy kolor i krawędzie poprawią wynik ponad semantykę oraz czy proponowana szczegółowa metoda będzie nowa.

Źródła pierwotne są podlinkowane przy twierdzeniach i w tabeli literatury. Główne lokalne materiały:

- [Przegląd commitów i reprodukcje problemów](REVIEW_SSSHANKS.md).
- [Adapter PixArt](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/lightning/dit_adapter.py).
- [Ewaluacja adaptera](C:/Users/maksi/AppData/Local/Temp/GSN_EEG_review_89142c2/eeg_mae/evaluation/eval_dit_adapter.py).
- [Obecna semantyka hierarchiczna](D:/GSN_EEG/eeg_mae/models/hierarchical_semantic.py).
- [Trening semantyki hierarchicznej](D:/GSN_EEG/eeg_mae/training/train_hierarchical_semantic_encoder.py).
- [Obecny FusionPrior](D:/GSN_EEG/eeg_mae/models/fusion_prior.py).
- [Raport koloru](D:/GSN_EEG/runs/eeg_color_encoder_direct/test_report.json), [raport krawędzi](D:/GSN_EEG/runs/eeg_edge_encoder_direct/test_report.json), [raport semantyki](D:/GSN_EEG/runs/eeg_hierarchical_encoder_eval/test_report.json).
