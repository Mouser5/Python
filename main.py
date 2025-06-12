import pandas as pd
import numpy as np

# 1. Cоздания Series
print("1. Series")

# Cписок
s1 = pd.Series([10, 20, 30])
print("1) Cписок:")
print(s1)


# Массив
s2 = pd.Series(np.array([1, 2, 3]))
print("\n2) Массив :")
print(s2)

# Скаляр
s3 = pd.Series(5, index=["a", "b", "c"])
print("\n3) Скаляр:")
print(s3)

# Словарь
s4 = pd.Series({"a": 100, "b": 200, "c": 300})
print("\n4) Словарь:")
print(s4)

# 2. Способы создания объектов DataFrame
print("\n2. DataFrame")

# Series
df1 = pd.DataFrame({"col1": pd.Series([1, 2, 3]), "col2": pd.Series([4, 5, 6])})
print("1) Series:")
print(df1)

# Списки словарей
df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
print("\n2) Списки словарей:")
print(df2)

# Словарь Series
s5=pd.Series([10,20])
s6=pd.Series([30,40])
df3 = pd.DataFrame({"col1": s5,
                    "col2": s6})
print("\n3) словарь Series:")
print(df3)

# Двумерный массив
df4 = pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["col1", "col2"])
print("\n4) 2D массив:")
print(df4)

# Структурированный массив
data = np.array([(1, "Alice"), (2, "Bob")], dtype=[("id", "i4"), ("name", "U10")])
df5 = pd.DataFrame(data)
print("\n5) Структурированный массив:")
print(df5)

# 3. Объединение Series
print("\n3. Объединение Series")

s1 = pd.Series({"a": 1, "b": 2})
s2 = pd.Series({"b": 3, "c": 4})

s_combined = s1.add(s2, fill_value=1) # замена на 1
print(s_combined)

# 4.транслирование для DataFrame
df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=['A', 'B', 'C'])
print("Исходный DataFrame:")
print(df)

# Вычитание среднего по столбцу
result = df - df.mean(axis=0)
print("\nПосле вычитания среднего по столбцам:")
print(result)

# 5. Методы ffill и bfill
print("\n5. Методы ffill() и bfill()")

df_nan = pd.DataFrame({
    "A": [np.nan, 2, np.nan, 4],
    "B": [1, np.nan, 3, np.nan]
})

print("DataFrame с пропущенными значениями:")
print(df_nan)

print("\nМетод ffill:")
print(df_nan.ffill())

print("\nМетод bfill:")
print(df_nan.bfill())
