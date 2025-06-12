import pandas as pd
import numpy as np

index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

population = [
    101,
    201,
    102,
    202,
    103,
    203
]

pop = pd.Series(population, index=index)
pop.index = pd.MultiIndex.from_tuples(index, names=['city', 'year'])

pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [10, 11, 12, 13, 14, 15]
    }
)

print("Начальные данные:\n")
print(pop_df)

print("\nВыбор данных для city_1 в столбце something\n")
pop_df_1 = pop_df.loc['city_1', 'something']
print(pop_df_1)

print("\nВыбор городов city_1, city_3 и столбцов total, something\n")
pop_df_2 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
print(pop_df_2)

print("\nВыбор городов city_1, city_3 в столбце something\n")
pop_df_3 = pop_df.loc[['city_1', 'city_3'], 'something']
print(pop_df_3)

# 2. Выборка по условиям

print("\n2. Выборки из DataFrame")

print("\nВсе данные за 2020 год:")
print(pop_df.xs(2020, level='year'))

# Создание DataFrame с job
index = pd.MultiIndex.from_product([
    ['city_1', 'city_2'],
    [2010, 2020]
], names=['city', 'year'])

columns = pd.MultiIndex.from_product([
    ['person_1', 'person_2', 'person_3'],
    ['job_1', 'job_2']
], names=['worker', 'job'])

data = np.arange(4 * 6).reshape(4, 6)
df = pd.DataFrame(data, index=index, columns=columns)

print("\nИзначальный DataFrame:")
print(df)

print("\nДанные по job_1:")
print(df.xs('job_1', axis=1, level='job'))

print("\nДанные по city_1 и job_2:")
print(df.loc['city_1'].xs('job_2', axis=1, level='job'))

# 3. Использование pd.IndexSlice
print("\n3. Использование IndexSlice")
idx = pd.IndexSlice

print("\nДанные по person_1 и person_3:")
print(df.loc[:, idx[["person_1", "person_3"], :]])

print("\nДанные по первому городу и первым двум person-ам:")
print(df.loc["city_1", idx["person_1":"person_2", :]])

# 4. inner, outer join
print("\n4. inner, outer join")


ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['b', 'c', 'f'], index=[3, 4, 5])


print("\nOuter join:")
print(pd.concat([ser1, ser2], axis=1, join='outer'))

print("\nInner join:")
print(pd.concat([ser1, ser2], axis=1, join='inner'))
