### Тема: Підбір правильних колокацій для сучасної англійської мови.

Хочу створити інструмент, що підказуватиме правильні колокації (verb + preposition, noun + adjective, etc.) під час написання тексту англійською, для людей з English as a second language.

В ідеалі, інструмент повинен виконувати дві основні функції:
1. Перевірити правильність колокацій у написаному тексті.
2. Запропонувати правильну колокацію з урахуванням контексту речення.


### Чому це важливо?

Англійська є мовою міжнародного спілкування: 1.5 млрд людей використовують англійську, в той же час рідною вона є лише для ~360 млн. людей. У багатьох non-native виникають складнощі з правильною колокацією, оскільки в англійскій мові вона може суттєво відрізнятися від рідної мови людини. Інструмент дозволить швидше писати вірно, а відтак - швидше опановувати мову.


### Які рішення вже існують?

Їх багато для вирішення обох задач.
Наприклад, для перевірки правильності колокацій можна скористатися:
1. Premium версією Grammarly;
2. ludwig.guru і подібні;
3. перевірка кількості веб-сторінок з колокацією в Google.

Для підбору вірних колокацій можна скористатися:
1. Словниками колокацій типу  Oxford Collocation Dictionary;
2. netspeak.org і подібні пошукові системи з n-grams.


### Чому мені потрібен новий інструмент?
1. доводиться перемикатися між різними програмами для підбору потрібних колокацій;
2. словники дають багато варіантів, доречність вживання в певному контексті доводиться визначати самостійно.


### Які дані планую використовувати?

A) Для вирішення задачі потрібен корпус сучасної англійської мови. Оскільки для мене важливим є використання англійської у діловому листуванні, планую зібрати корпус із текстів ділових англомовних видань з гарантовано високоякісною англійською:
1. The New York Times (us)
2. Financial Times (gb)
3. The Economist (gb)
4. The Wall Street Journal (us)
5. The Telegraph (gb)
6. The Huffington Post (us)
7. Bloomberg (us)
...

B) Для прикладів помилок, що їх припускаються люди з English as a Second Language, планую скористатися:
1. [NUCLE corpus](https://cogcomp.org/page/resource_view/57)
2. [Lang-8 Learner Corpora](http://cl.naist.jp/nldata/lang-8/)


### Очікуваний формат рішення

1. Користувач вводить речення.
2. Алгоритм аналізує правильність колокацій, інформує користувача про помилку, якщо така буде.
3. Користувач обирає слово у реченні для якого хоче побачити можливі колокації.
Це може бути як слово з помилкової колокації, так і будь-яке інше слово, до якого потрібно підібрати колокацію (наприклад, прикметник до іменника)
4. Для обраного користувачем слова відображаються колокації, які зустрічаються в корпусі згруаовані за частиною мови/типом зв'язку.
5. Користувач обирає один з можливих варіантів колокації і отримує кілька прикладів вживання з корпусу текстів.


### Базове рішення

Базовим рішенням буде пошук проблемних (невірних) колокацій у введеному тексті та пропозиція про заміну на найбільш вживану колокацію для головного слова.



