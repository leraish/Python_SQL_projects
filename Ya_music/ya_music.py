#!/usr/bin/env python
# coding: utf-8

# # Яндекс Музыка

# Сравнение Москвы и Петербурга окружено мифами. Например:
#  * Москва — мегаполис, подчинённый жёсткому ритму рабочей недели;
#  * Петербург — культурная столица, со своими вкусами.
# 
# На данных Яндекс Музыки вы сравните поведение пользователей двух столиц.
# 
# **Цель исследования** — проверьте три гипотезы:
# 1. Активность пользователей зависит от дня недели. Причём в Москве и Петербурге это проявляется по-разному.
# 2. В понедельник утром в Москве преобладают одни жанры, а в Петербурге — другие. Так же и вечером пятницы преобладают разные жанры — в зависимости от города. 
# 3. Москва и Петербург предпочитают разные жанры музыки. В Москве чаще слушают поп-музыку, в Петербурге — русский рэп.
# 
# **Ход исследования**
# 
# Данные о поведении пользователей вы получите из файла `yandex_music_project.csv`. О качестве данных ничего не известно. Поэтому перед проверкой гипотез понадобится обзор данных. 
# 
# Вы проверите данные на ошибки и оцените их влияние на исследование. Затем, на этапе предобработки вы поищете возможность исправить самые критичные ошибки данных.
#  
# Таким образом, исследование пройдёт в три этапа:
#  1. Обзор данных.
#  2. Предобработка данных.
#  3. Проверка гипотез.
# 
# 

# ## Обзор данных
# 
# Составьте первое представление о данных Яндекс Музыки.
# 
# 
# 

# **Задание 1**
# 
# Основной инструмент аналитика — `pandas`. Импортируйте эту библиотеку.

# In[1]:


import pandas as pd


# **Задание 2**
# 
# Прочитайте файл `yandex_music_project.csv` из папки `/datasets` и сохраните его в переменной `df`:

# In[2]:


df = pd.read_csv('/datasets/yandex_music_project.csv')


# **Задание 3**
# 
# 
# Выведите на экран первые десять строк таблицы:

# In[3]:


df.head(10)


# **Задание 4**
# 
# 
# Одной командой получить общую информацию о таблице c помощью метода `info()`:

# In[4]:


df.info()


# Итак, в таблице семь столбцов. Тип данных во всех столбцах — `object`.
# 
# Согласно документации к данным:
# * `userID` — идентификатор пользователя;
# * `Track` — название трека;  
# * `artist` — имя исполнителя;
# * `genre` — название жанра;
# * `City` — город пользователя;
# * `time` — время начала прослушивания;
# * `Day` — день недели.
# 
# Количество значений в столбцах различается. Значит, в данных есть пропущенные значения.

# **Задание 5**
# 
# **Вопрос со свободной формой ответа**
# 
# В названиях колонок видны нарушения стиля:
# * Строчные буквы сочетаются с прописными.
# * Встречаются пробелы.
# 
# Какое третье нарушение?

# In[5]:


# не соблюден змеиный_регистр для столбца userID


# **Выводы**
# 
# В каждой строке таблицы — данные о прослушанном треке. Часть колонок описывает саму композицию: название, исполнителя и жанр. Остальные данные рассказывают о пользователе: из какого он города, когда он слушал музыку. 
# 
# Предварительно можно утверждать, что данных достаточно для проверки гипотез. Но встречаются пропуски в данных, а в названиях колонок — расхождения с хорошим стилем.
# 
# Чтобы двигаться дальше, нужно устранить проблемы в данных.

# ## Предобработка данных
# Исправьте стиль в заголовках столбцов, исключите пропуски. Затем проверьте данные на дубликаты.

# ### Стиль заголовков
# 
# **Задание 6**
# 
# Выведите на экран названия столбцов:

# In[6]:


# перечень названий столбцов таблицы df
df.columns


# **Задание 7**
# 
# 
# Приведите названия в соответствие с хорошим стилем:
# * несколько слов в названии запишите в «змеином_регистре»,
# * все символы сделайте строчными,
# * устраните пробелы.
# 
# Для этого переименуйте колонки так:
# * `'  userID'` → `'user_id'`;
# * `'Track'` → `'track'`;
# * `'  City  '` → `'city'`;
# * `'Day'` → `'day'`.

# In[7]:


# переименование столбцов
df = df.rename(columns = {'  userID' : 'user_id', 'Track' : 'Track'.lower(),'  City  ' : 'city','Day': 'Day'.lower()})


# **Задание 8**
# 
# 
# Проверьте результат. Для этого ещё раз выведите на экран названия столбцов:

# In[8]:


# проверка результатов - перечень названий столбцов
df.columns


# ### Пропуски значений
# 
# **Задание 9**
# 
# Сначала посчитайте, сколько в таблице пропущенных значений. Для этого достаточно двух методов `pandas`:

# In[9]:


# подсчёт пропусков
df.isna().sum()


# Не все пропущенные значения влияют на исследование. Так в `track` и `artist` пропуски не важны для вашей работы. Достаточно заменить их явными обозначениями.
# 
# Но пропуски в `genre` могут помешать сравнению музыкальных вкусов в Москве и Санкт-Петербурге. На практике было бы правильно установить причину пропусков и восстановить данные. Такой возможности нет в учебном проекте. Придётся:
# * заполнить и эти пропуски явными обозначениями;
# * оценить, насколько они повредят расчётам. 

# **Задание 10**
# 
# Замените пропущенные значения в столбцах `track`, `artist` и `genre` на строку `'unknown'`. Для этого создайте список `columns_to_replace`, переберите его элементы циклом `for` и для каждого столбца выполните замену пропущенных значений:

# In[10]:


# перебор названий столбцов в цикле и замена пропущенных значений на 'unknown'
columns_to_replace = ['track', 'artist', 'genre']
for column in columns_to_replace:
    df[column] = df[column].fillna('unknown')


# **Задание 11**
# 
# Убедитесь, что в таблице не осталось пропусков. Для этого ещё раз посчитайте пропущенные значения.

# In[11]:


# подсчёт пропусков
df.isna().sum()


# ### Дубликаты
# 
# **Задание 12**
# 
# Посчитайте явные дубликаты в таблице одной командой:

# In[12]:


# подсчёт явных дубликатов
df.duplicated().sum()


# **Задание 13**
# 
# Вызовите специальный метод `pandas`, чтобы удалить явные дубликаты:

# In[13]:


# удаление явных дубликатов
df = df.drop_duplicates()


# **Задание 14**
# 
# Ещё раз посчитайте явные дубликаты в таблице — убедитесь, что полностью от них избавились:

# In[14]:


# проверка на отсутствие дубликатов
df.duplicated().sum()


# Теперь избавьтесь от неявных дубликатов в колонке `genre`. Например, название одного и того же жанра может быть записано немного по-разному. Такие ошибки тоже повлияют на результат исследования.

# **Задание 15**
# 
# Выведите на экран список уникальных названий жанров, отсортированный в алфавитном порядке. Для этого:
# 1. извлеките нужный столбец датафрейма; 
# 2. примените к нему метод сортировки;
# 3. для отсортированного столбца вызовите метод, который вернёт уникальные значения из столбца.

# In[15]:


# Просмотр уникальных названий жанров
genres = df.sort_values(by='genre')
genres['genre'].unique()


# **Задание 16**
# 
# Просмотрите список и найдите неявные дубликаты названия `hiphop`. Это могут быть названия с ошибками или альтернативные названия того же жанра.
# 
# Вы увидите следующие неявные дубликаты:
# * *hip*,
# * *hop*,
# * *hip-hop*.
# 
# Чтобы очистить от них таблицу используйте метод `replace()` с двумя аргументами: списком строк-дубликатов (включащий *hip*, *hop* и *hip-hop*) и строкой с правильным значением. Вам нужно исправить колонку `genre` в таблице `df`: заменить каждое значение из списка дубликатов на верное. Вместо `hip`, `hop` и `hip-hop` в таблице должно быть значение `hiphop`:

# In[48]:


# Устранение неявных дубликатов
# Список неявных дубликатов жанра hiphop:
hiphop_duplicates = ['hip', 'hop', 'hip-hop']
#Корректное название жанра, на которое следует заменить дубликаты:
name = 'hiphop'

df['genre'] = df['genre'].replace(hiphop_duplicates, name)


# **Задание 17**
# 
# Проверьте, что заменили неправильные названия:
# 
# *   hip,
# *   hop,
# *   hip-hop.
# 
# Выведите отсортированный список уникальных значений столбца `genre`:

# In[44]:


# Проверка на неявные дубликаты
genres = df.sort_values(by='genre')
genres['genre'].unique()


# **Выводы**
# 
# Предобработка обнаружила три проблемы в данных:
# 
# - нарушения в стиле заголовков,
# - пропущенные значения,
# - дубликаты — явные и неявные.
# 
# Вы исправили заголовки, чтобы упростить работу с таблицей. Без дубликатов исследование станет более точным.
# 
# Пропущенные значения вы заменили на `'unknown'`. Ещё предстоит увидеть, не повредят ли исследованию пропуски в колонке `genre`.
# 
# Теперь можно перейти к проверке гипотез. 

# ## Проверка гипотез

# ### Сравнение поведения пользователей двух столиц

# Первая гипотеза утверждает, что пользователи по-разному слушают музыку в Москве и Санкт-Петербурге. Проверьте это предположение по данным о трёх днях недели — понедельнике, среде и пятнице. Для этого:
# 
# * Разделите пользователей Москвы и Санкт-Петербурга.
# * Сравните, сколько треков послушала каждая группа пользователей в понедельник, среду и пятницу.
# 

# **Задание 18**
# 
# Для тренировки сначала выполните каждый из расчётов по отдельности. 
# 
# Оцените активность пользователей в каждом городе. Сгруппируйте данные по городу и посчитайте прослушивания в каждой группе.
# 
# 

# In[18]:


# Подсчёт прослушиваний в каждом городе

cities_play_count = df.groupby('city')['city'].count()
cities_play_count


# В Москве прослушиваний больше, чем в Петербурге. Из этого не следует, что московские пользователи чаще слушают музыку. Просто самих пользователей в Москве больше.
# 
# **Задание 19**
# 
# Теперь сгруппируйте данные по дню недели и посчитайте прослушивания в понедельник, среду и пятницу. Учтите, что в данных есть информация о прослушиваниях только за эти дни.
# 

# In[19]:


# Подсчёт прослушиваний в каждый из трёх дней
grouping_by_days = df.groupby('day')['city'].count()
grouping_by_days


# В среднем пользователи из двух городов менее активны по средам. Но картина может измениться, если рассмотреть каждый город в отдельности.

# **Задание 20**
# 
# 
# Вы видели, как работает группировка по городу и по дням недели. Теперь напишите функцию, которая объединит два эти расчёта.
# 
# Создайте функцию `number_tracks()`, которая посчитает прослушивания для заданного дня и города. Ей понадобятся два параметра:
# * день недели,
# * название города.
# 
# В функции сохраните в переменную строки исходной таблицы, у которых значение:
#   * в колонке `day` равно параметру `day`,
#   * в колонке `city` равно параметру `city`.
# 
# Для этого примените последовательную фильтрацию с логической индексацией (или сложные логические выражения в одну строку, если вы уже знакомы с ними).
# 
# Затем посчитайте значения в столбце `user_id` получившейся таблицы. Результат сохраните в новую переменную. Верните эту переменную из функции.

# In[20]:


# <создание функции number_tracks()>
# Объявляется функция с двумя параметрами: day, city.
def number_tracks(day, city):
    
# В переменной track_list сохраняются те строки таблицы df, для которых 
# значение в столбце 'day' равно параметру day и одновременно значение
# в столбце 'city' равно параметру city (используйте последовательную фильтрацию
# с помощью логической индексации или сложные логические выражения в одну строку, если вы уже знакомы с ними).

    track_list = df[df['day'] == day]
    track_list = track_list[track_list['city'] == city]
# В переменной track_list_count сохраняется число значений столбца 'user_id',
# рассчитанное методом count() для таблицы track_list.
    track_list_count = track_list['user_id'].count()
# Функция возвращает число - значение track_list_count.
    return track_list_count

    


# Функция для подсчёта прослушиваний для конкретного города и дня.
# С помощью последовательной фильтрации с логической индексацией она 
# сначала получит из исходной таблицы строки с нужным днём,
# затем из результата отфильтрует строки с нужным городом,
# методом count() посчитает количество значений в колонке user_id. 
# Это количество функция вернёт в качестве результата


# **Задание 21**
# 
# Вызовите `number_tracks()` шесть раз, меняя значение параметров — так, чтобы получить данные для каждого города в каждый из трёх дней.

# In[21]:


# количество прослушиваний в Москве по понедельникам
number_tracks('Monday', 'Moscow')


# In[22]:


# количество прослушиваний в Санкт-Петербурге по понедельникам
number_tracks('Monday', 'Saint-Petersburg')


# In[23]:


# количество прослушиваний в Москве по средам
number_tracks('Wednesday', 'Moscow')


# In[24]:


# количество прослушиваний в Санкт-Петербурге по средам
number_tracks('Wednesday', 'Saint-Petersburg')


# In[25]:


# количество прослушиваний в Москве по пятницам
number_tracks('Friday', 'Moscow')


# In[26]:


# количество прослушиваний в Санкт-Петербурге по пятницам
number_tracks('Friday', 'Saint-Petersburg')


# **Задание 22**
# 
# Создайте c помощью конструктора `pd.DataFrame` таблицу, где
# * названия колонок — `['city', 'monday', 'wednesday', 'friday']`;
# * данные — результаты, которые вы получили с помощью `number_tracks`.

# In[27]:


# Таблица с результатами
# Данные для DataFrame:
data = [
    ['Moscow', 15740, 11056, 15945],
    ['Saint-Petersburg', 5614, 7003, 5895]
    ]
# Названия колонок:
columns = ['city', 'monday', 'wednesday', 'friday']

pd.DataFrame(data = data, columns = columns)


# **Выводы**
# 
# Данные показывают разницу поведения пользователей:
# 
# - В Москве пик прослушиваний приходится на понедельник и пятницу, а в среду заметен спад.
# - В Петербурге, наоборот, больше слушают музыку по средам. Активность в понедельник и пятницу здесь почти в равной мере уступает среде.
# 
# Значит, данные говорят в пользу первой гипотезы.

# ### Музыка в начале и в конце недели

# Согласно второй гипотезе, утром в понедельник в Москве преобладают одни жанры, а в Петербурге — другие. Так же и вечером пятницы преобладают разные жанры — в зависимости от города.

# **Задание 23**
# 
# Сохраните таблицы с данными в две переменные:
# * по Москве — в `moscow_general`;
# * по Санкт-Петербургу — в `spb_general`.

# In[28]:


# получение таблицы moscow_general из тех строк таблицы df, 
# для которых значение в столбце 'city' равно 'Moscow'
moscow_general = df[df['city'] == 'Moscow']


# In[29]:


# получение таблицы spb_general из тех строк таблицы df,
# для которых значение в столбце 'city' равно 'Saint-Petersburg'
spb_general = df[df['city'] == 'Saint-Petersburg']
spb_general


# **Задание 24**
# 
# Создайте функцию `genre_weekday()` с четырьмя параметрами:
# * таблица (датафрейм) с данными,
# * день недели,
# * начальная временная метка в формате 'hh:mm', 
# * последняя временная метка в формате 'hh:mm'.
# 
# Функция должна вернуть информацию о топ-10 жанров тех треков, которые прослушивали в указанный день, в промежутке между двумя отметками времени.

# In[30]:


# Объявление функции genre_weekday() с параметрами table, day, time1, time2,
# которая возвращает информацию о самых популярных жанрах в указанный день в
# заданное время:
# 1) в переменную genre_df сохраняются те строки переданного датафрейма table, для
#    которых одновременно:
#    - значение в столбце day равно значению аргумента day
#    - значение в столбце time больше значения аргумента time1
#    - значение в столбце time меньше значения аргумента time2
#    Используйте последовательную фильтрацию с помощью логической индексации.
# 2) сгруппировать датафрейм genre_df по столбцу genre, взять один из его
#    столбцов и посчитать методом count() количество записей для каждого из
#    присутствующих жанров, получившийся Series записать в переменную
#    genre_df_count
# 3) отсортировать genre_df_count по убыванию встречаемости и сохранить
#    в переменную genre_df_sorted
# 4) вернуть Series из 10 первых значений genre_df_sorted, это будут топ-10
#    популярных жанров (в указанный день, в заданное время)

def genre_weekday(table, day, time1, time2):
    # последовательная фильтрация
    # оставляем в genre_df только те строки df, у которых день равен day
    genre_df = table[table['day'] == day]
    
    # оставляем в genre_df только те строки genre_df, у которых время меньше time2
    genre_df = genre_df[genre_df['time'] < time2]
    
    # оставляем в genre_df только те строки genre_df, у которых время больше time1
    genre_df = genre_df[genre_df['time'] > time1]
    
    # сгруппируем отфильтрованный датафрейм по столбцу с названиями жанров, возьмём столбец genre и посчитаем кол-во строк для каждого жанра методом count()
    genre_df_count =  genre_df.groupby('genre')['genre'].count()
   
    
    # отсортируем результат по убыванию (чтобы в начале Series оказались самые популярные жанры)
    genre_df_sorted = genre_df_count.sort_values(ascending=False)
    
    # вернём Series с 10 самыми популярными жанрами в указанный отрезок времени заданного дня
    return genre_df_sorted[:10]


# **Задание 25**
# 
# 
# Cравните результаты функции `genre_weekday()` для Москвы и Санкт-Петербурга в понедельник утром (с 7:00 до 11:00) и в пятницу вечером (с 17:00 до 23:00):

# In[31]:


# вызов функции для утра понедельника в Москве (вместо df — таблица moscow_general)
# объекты, хранящие время, являются строками и сравниваются как строки
# пример вызова: genre_weekday(moscow_general, 'Monday', '07:00', '11:00')

genre_weekday(moscow_general, 'Monday', '07:00', '11:00')


# In[32]:


# вызов функции для утра понедельника в Петербурге (вместо df — таблица spb_general)
genre_weekday(spb_general, 'Monday', '07:00', '11:00')


# In[33]:


# вызов функции для вечера пятницы в Москве
genre_weekday(moscow_general, 'Friday', '17:00', '23:00')


# In[34]:


# вызов функции для вечера пятницы в Петербурге
genre_weekday(spb_general, 'Friday', '17:00', '23:00')


# **Выводы**
# 
# Если сравнить топ-10 жанров в понедельник утром, можно сделать такие выводы:
# 
# 1. В Москве и Петербурге слушают похожую музыку. Единственное отличие — в московский рейтинг вошёл жанр “world”, а в петербургский — джаз и классика.
# 
# 2. В Москве пропущенных значений оказалось так много, что значение `'unknown'` заняло десятое место среди самых популярных жанров. Значит, пропущенные значения занимают существенную долю в данных и угрожают достоверности исследования.
# 
# Вечер пятницы не меняет эту картину. Некоторые жанры поднимаются немного выше, другие спускаются, но в целом топ-10 остаётся тем же самым.
# 
# Таким образом, вторая гипотеза подтвердилась лишь частично:
# * Пользователи слушают похожую музыку в начале недели и в конце.
# * Разница между Москвой и Петербургом не слишком выражена. В Москве чаще слушают русскую популярную музыку, в Петербурге — джаз.
# 
# Однако пропуски в данных ставят под сомнение этот результат. В Москве их так много, что рейтинг топ-10 мог бы выглядеть иначе, если бы не утерянные  данные о жанрах.

# ### Жанровые предпочтения в Москве и Петербурге
# 
# Гипотеза: Петербург — столица рэпа, музыку этого жанра там слушают чаще, чем в Москве.  А Москва — город контрастов, в котором, тем не менее, преобладает поп-музыка.

# **Задание 26**
# 
# Сгруппируйте таблицу `moscow_general` по жанру и посчитайте прослушивания треков каждого жанра методом `count()`. Затем отсортируйте результат в порядке убывания и сохраните его в таблице `moscow_genres`.

# In[35]:


# одной строкой: группировка таблицы moscow_general по столбцу 'genre', 
# подсчёт числа значений 'genre' в этой группировке методом count(), 
# сортировка получившегося Series в порядке убывания и сохранение в moscow_genres

moscow_genres = moscow_general.groupby('genre')['genre'].count().sort_values(ascending=False)


# **Задание 27**
# 
# Выведите на экран первые десять строк `moscow_genres`:

# In[36]:


# просмотр первых 10 строк moscow_genres
moscow_genres.head(10)


# **Задание 28**
# 
# 
# Теперь повторите то же и для Петербурга.
# 
# Сгруппируйте таблицу `spb_general` по жанру. Посчитайте прослушивания треков каждого жанра. Результат отсортируйте в порядке убывания и сохраните в таблице `spb_genres`:
# 

# In[37]:


# одной строкой: группировка таблицы spb_general по столбцу 'genre', 
# подсчёт числа значений 'genre' в этой группировке методом count(), 
# сортировка получившегося Series в порядке убывания и сохранение в spb_genres
spb_genres = spb_general.groupby('genre')['genre'].count().sort_values(ascending=False)


# **Задание 29**
# 
# Выведите на экран первые десять строк `spb_genres`:

# In[38]:


spb_genres.head(10)


# **Выводы**

# Гипотеза частично подтвердилась:
# * Поп-музыка — самый популярный жанр в Москве, как и предполагала гипотеза. Более того, в топ-10 жанров встречается близкий жанр — русская популярная музыка.
# * Вопреки ожиданиям, рэп одинаково популярен в Москве и Петербурге. 
# 

# ## Итоги исследования

# Вы проверили три гипотезы и установили:
# 
# 1. День недели по-разному влияет на активность пользователей в Москве и Петербурге. 
# 
# Первая гипотеза полностью подтвердилась.
# 
# 2. Музыкальные предпочтения не сильно меняются в течение недели — будь то Москва или Петербург. Небольшие различия заметны в начале недели, по понедельникам:
# * в Москве слушают музыку жанра “world”,
# * в Петербурге — джаз и классику.
# 
# Таким образом, вторая гипотеза подтвердилась лишь отчасти. Этот результат мог оказаться иным, если бы не пропуски в данных.
# 
# 3. Во вкусах пользователей Москвы и Петербурга больше общего чем различий. Вопреки ожиданиям, предпочтения жанров в Петербурге напоминают московские.
# 
# Третья гипотеза не подтвердилась. Если различия в предпочтениях и существуют, на основной массе пользователей они незаметны.
# 
# **На практике исследования содержат проверки статистических гипотез.**
# Из данных одного сервиса не всегда можно сделать вывод о всех жителях города.
# Проверки статистических гипотез покажут, насколько они достоверны, исходя из имеющихся данных. 
# С методами проверок гипотез вы ещё познакомитесь в следующих темах.