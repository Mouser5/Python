import numpy as np
#1
a = np.ones((3, 2)).reshape(2, 3) #поменять форму массива a
b = np.arange(3)
c = a + b
print(c, c.shape)

#2
y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(np.sum((y > 3) & (y < 9), axis=0))
print(np.sum((y > 3) & (y < 9), axis=1))
