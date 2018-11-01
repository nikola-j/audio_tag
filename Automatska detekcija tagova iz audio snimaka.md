# Automatska detekcija tagova iz audio snimaka



## Vggish model

VGGish je model koji je istreniran od strane Googleove istraživačke grupe za klasifikaciju zvukova nad youtube snimcima. To je ustvari adaptirani VGG16 model koji je modifikovan da se nosi sa zvucnim snimcima tako sto mu je na ulaz umesto slike stavljen spektogram velicine 96 x 64 koji odgovara 960 milisekundi zvuka a na izlaz zatrazena labela zvucnog signala. 

| ![img](.images/vgg16.png) |
| :--: |
| VGG16 arhitektura |

Za trening su koristili 70 miliona youtube snimaka, kada ih podele na snimke dužine 960 milisekundi dobiju 20 biliona uzoraka za trening. Nad svakim uzorkom primene Furijeovu transformaciju sa prozorima dužine 25ms i prozorom svakih 10ms. Dobijen spektrogram se integriše u 64 binova koji su mel razmaknuti, i na magnitudu svakog bina je primenjena logaritamska transformacija, čime dobiju ulazne "slike" veličine 96 x 64.

| ![img](.images/1541068550301.png) | ![1541068587234](.images/1541068587234.png) | ![1541068607465](.images/1541068607465.png) |
| :--: | :--: |  :--: |
| Zvučni signal bubnjeva | Isti signal posle Furijeove transformacije | Isti signal posle mel binovanja |

