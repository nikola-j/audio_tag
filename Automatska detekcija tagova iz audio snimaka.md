# Automatska detekcija tagova iz audio snimaka

## Uvod

Jedan od najvećih problema sa kojim se suočava semantički veb jeste tagovanje. Pošto korisnici interneta pri kreiranju sadržaja se retko kada posvete tagovanju materijala, jedino skalabilno rešenje jeste da se to radi automatski. Za automatsko tagovanje teksta postoje razne tehnike (PMI, TF-IDF itd), ali za snimke zvuka je to jako teško raditi jer podaci imaju mnogo više šuma. Ali u zadnjih nekoliko godina sistemi koji rešavaju ovaj problem imaju velik porast u performansama zbog povećanog korišćenja neuralnih mreža i sve većih setova podataka.

| ![1541149013758](.images/1541149013758.png) |
| :----------------------------------------------------------: |
| Performanse sistema za detekciju događaja tokom vremena ([izvor](http://www.cs.tut.fi/~heittolt/research-sound-event-detection)) |

U ovom radu ću opisati moj sistem za automatsko tagovanje zvuka koji koristi prenos učenja sa mnogo većeg modela  (VGGish) koji je istreniran nad mnogo podataka (oko 70 miliona jutub klipova) na manji set podataka koji ima  drugačije labele.

## Vggish model

VGGish je model koji je istreniran od strane Guglove istraživačke grupe za klasifikaciju zvukova nad jutjub snimcima. To je ustvari adaptirani VGG16 model koji je modifikovan da se nosi sa zvučnim snimcima tako što mu je na ulaz umesto slike stavljen spektogram velicine 96 x 64 koji odgovara 960 milisekundi zvuka a na izlaz zatrazena jedna od 527 labela zvucnog signala.

| ![img](.images/vgg16.png) |
| :--: |
| VGG16 arhitektura |

Za trening su koristili 70 miliona jutjub snimaka, kada ih podele na snimke dužine 960 milisekundi dobiju 20 biliona uzoraka za trening. Nad svakim uzorkom primene Furijeovu transformaciju sa prozorima dužine 25ms i prozorom svakih 10ms. Dobijen spektrogram se integriše u 64 binova koji su mel razmaknuti, i na magnitudu svakog bina je primenjena logaritamska transformacija, čime dobiju ulazne "slike" veličine 96 x 64.

| ![img](.images/1541068550301.png) | ![1541068587234](.images/1541068587234.png) | ![1541068607465](.images/1541068607465.png) |
| :--: | :--: |  :--: |
| Zvučni signal bubnjeva | Isti signal posle Furijeove transformacije | Isti signal posle mel binovanja |

Ovako istreniranom modelu oni su izbacili zadnjih par zadnjih slojeva nakon treninga. Čime on za svaku sekundu (960ms) zvuka na ulazu predviđa vektor dužine 128 koji "semantički" opisuje signal na ulazu. Time se dobija model koji je odličan za  prenošenje iskustva na nove zadatke. Google je objavio ovaj model i on se može skinuti [ovde](https://github.com/tensorflow/models/tree/master/research/audioset).

## Prenos iskustva

U dubokom učenju prenos iskustva označava korišćenje nekog modela za brže treniranje drugog modela na sličnim zadacima. Ovo se radi jer treniranje modela nad velikim setom podataka traje jako dugo i košta mnogo kompjuterskih resorsa. Korišćenjem takvog pre-treniranog modela omogućava bolje performanse na novom zadatku ukoliko   zadatak ima veze sa originalnim zadatkom.

U ovom slučaju koristio sam pre-trenirani VGGish model kako bih olakšao trening nad [ovim](https://www.kaggle.com/c/freesound-audio-tagging/) setom podataka koji ima samo 9400 zvučnih uzoraka, gde je za svaki uzorak potrebno predvideti pravu labelu (jednu od 41). Set podataka sam preprocesirao i provukao kroz VGGish model, čime sam dobio svojstva ulaznog signala zapisane kao 128 brojeva za svaku sekundu.

Onda sam trenirao nekoliko modela nad tim novim setom podataka čime sam dobio model koji u 55% slučajeva predvidi tačnu labelu za sekundu snimka, a u 81% slučajeva je prava labela među prvih pet predviđanja.

| <img src=".images/modelsnn.png" alt="drawing" width="300"/> |
| :----------------------------------------------------------: |
| Model koji je dostigao navedene performanse |

## Zaključak

Prenos iskustva je odlična tehnika kada je potrebno rešiti neki specifični zadatak, a nemamo dovoljno kompjuterskih resorsa ili nemamo veliki set podataka.

Napretkom dubokog učenja automatsko procesiranje velikog broja podataka koje stvara internet postaje sve bolje i bolje, verujem da će to najviše doprineti da se koncepti semantičkog veba više primenjuju nego sada.