*Тема проекту* -  визначення характеристик людини що написала текст, зокрема психотипу.

Шкала психотипів за типологією Майерса—Бріггса(екстраверт\інтроверт, судження\перцепція, комбінації, тощо.
Абревіатури типів -  ENFJ, INFP і тд. ). 
В англомовному інтернеті ця шкала досить популярна, так у статті https://nlp.stanford.edu/courses/cs224n/2015/reports/6.pdf 
автори знаходять твіти тих користувачів у яких є згадки про типи, і виділяють серед них ті, які несуть інформацію про психотип
користувача, наприклад *"@ProfCarol Just wondering, what’s your type? I’m an ENFJ"*, і далі аналізують їх останні пости. 
Так вони створили базу в 90 тисяч юзерів з їх твітами!

Зрозуміло, що для побудови такої бази треба в першу чергу навчитися відкидати  випадки коли психотипи згадуються у неправильному контексті, або ці абревіатури значать щось інше.
Далі з повідомлень вибраних користувачів виокремлюються фічі, розв'язується стандартна задача класифікації.
Цінність цього дослідження я вбачаю не тільки у самій класифікації, а і у тих залежностях які можуть бути встановлені.
Як наприклад цікава залежність із статті - екстраверти частіше використовують хештеги.
У якості бейслайну якраз і можна взяти\придумати набір регулярних виразів для перевірки таких залежностей.

Головна складність в отриманні даних та їх попередній обробці, треба розбиратися і штурмувати твітер апі, схоже дані за період більше 30-днів не безкоштовні. 
Можливо вдасться знайти й інші джерела, іншою мовою.
Було б круто позмагатися з результатами із зазначеної статі, можливо вдасться визначати додаткові характеристики, стать, вік тощо.