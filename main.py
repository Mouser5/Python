import matplotlib.pyplot as plt
import numpy as np

x1=np.linspace(0, 20,5)
y1=np.array([1,7,3,5,11])
y2=np.array([4,3,1,8,12])
plt.plot(x1, y1, '-r.')
plt.plot(x1, y2, '-.g.')
plt.legend(("line 1","line 1"))
plt.show()

x2 = np.linspace(1,5,5)
y1 = np.array([1, 7, 6, 3, 5])
plt.subplot(211)
plt.plot(x2,y1)

plt.subplot(223)
plt.plot(x2,np.pow(x2-3,2)*2+2)

plt.subplot(224)
y2=np.array(([-7,-4,2,-4,-7]))
plt.plot(x2,y2)

plt.show()

x3=np.arange(-4,4,0.2)
y3=np.pow(x3,2)
plt.annotate('min',xy=(0,0),xytext=(0,11),arrowprops=dict(facecolor='green', shrink=0.05))
plt.plot(x3,y3)
plt.show()

np.random.seed(19680801)
data=np.random.randint(11,size=(7,7))
plt.pcolormesh(data)
plt.colorbar()
plt.show()

x = np.arange(0.0, 5, 0.01)
y = np.cos(x*np.pi)
plt.plot(x, y, c = "r")
plt.fill_between(x, y)
plt.show()

x = np.arange(0.0, 5, 0.01)
y = np.cos(x*np.pi)
y_masked = np.ma.masked_where(y < -0.5, y)
plt.ylim(-1, 1)
plt.plot(x, y_masked, linewidth=3)
plt.show()


x = np.arange(0, 7)
y = x
where_set = ['pre', 'post', 'mid']
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axs):
  ax.step(x, y, "g-o", where=where_set[i])
  ax.grid()

plt.show()

x = np.arange(0, 11, 1)

y1 = np.array([(-0.2)*i**2+2*i for i in x])
y2 = np.array([(-0.4)*i**2+4*i for i in x])
y3 = np.array([2*i for i in x])

labels = ["y1", "y2", "y3"]

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3, labels=labels)
ax.legend(loc='upper left')
plt.show()


vals = [ 22, 16, 50, 20, 37]
labels = ["Ford", "Toyota", "BMV", "AUDI", "Jaguar"]
fig, ax = plt.subplots()
explode = (0, 0, 0.2, 0, 0)
ax.pie(vals, labels=labels, explode=explode)
ax.axis("equal")
plt.show()

vals = [ 22, 16, 50, 20, 37]
labels = ["Ford", "Toyota", "BMV", "AUDI", "Jaguar"]
fig, ax = plt.subplots()
ax.pie(vals, labels=labels, wedgeprops=dict(width=0.5))
ax.axis("equal")
plt.show()
