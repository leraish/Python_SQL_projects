# Исследование рынка заведений общественного питания Москвы. Открытие кофейни

---- 

Инвесторы из фонда «Shut Up and Take My Money» нацелены на открытие кофейни в Москве. Необходимо исследовать рынок и составить рекомендации для открытия заведения.

**Задачи:** исследовательский анализ рынка общепита Москвы, рекомендации на основе полученных выводов, составление презентации

----

**Инструменты и навыки:** Python, Pandas, Matplotlib, Plotly, Json, Choropleth, визуализация данных, презентации

----

**Выводы**

С ходом исследования, развернутыми выводами, рассчетами и визуализациями можно ознакомиться в файле .ipynb. Из интересных наблюдений:
* 38.1% заведений - сетевые, из них больше половины - кофейни, с которыми нам предстоит конкурировать. Сетевые заведения имеют низкий относительно несетевых средний чек и распространяют свои заведения на все округи Москвы - конкурировать с сетями достаточно сложно
* у заведений ЦАО и ЗАО ожидаемо более высокий средний чек, в этих же округах средний рейтинг заведений выше; а в ЮВАО и ВАО наоборот -  заведения с низкими рейтингами чаще располагаются именно там
* Численно кофеен больше в ЦАО и САО, доли кофеен среди остальных заведений внутри округа ~20% у каждого

**Для открытия новой кофейни рекомендуется ориентироваться на ЦАО:** самый популярный округ, высокий средний чек и рейтинг завдений. 

Основные ориентиры для открытии:
* стоимость чашки кофе: 139-250 руб
* средний чек: 350-1137 (медиана 500, отталкиваемся от формата будущей кофейни)
* 40-143 посадочных мест (опять же отталкиваемся от формата заведения и размера помещения)

Территориальные ориентиры:
* улицы, на которых расположено много заведений (улица популярна), но при этом среди заведений не более одной кофейни (избежим острой конкуренции)
* отобраны улицы с вышеописанными характеристиками + условие: либо у заведений на этой улице высокий средний чек, либо высокий средний рейтинг. Рекомендуется открыть заведение на отобранных или примыкающим к ним улицах


----

*Комментарий:* интерактивные графики plotly не отображаются в hithub: в тетрадке настроено отоброжение графиков в формате png 

