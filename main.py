import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Загрузка и фильтрация данных
iris = px.data.iris()
filtered_data = iris[iris['species'].isin(['setosa', 'virginica'])]

X = filtered_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = filtered_data['species'].values

# Кодировка классов в -1 и 1
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y = np.where(y_encoded == 0, -1, 1)

# PCA для снижения размерности до 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Реализация метода опорных векторов
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_i = y[idx]
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - y_i * x_i
                    db = -y_i
                self.w -= self.lr * dw
                self.b -= self.lr * db
        return self.w, self.b

# Обучение SVM и расчет разделяющей плоскости
model = SVM()
coef, intercept = model.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

if coef[2] != 0:
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
else:
    zz = np.zeros_like(xx)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], mode='markers',
                           marker=dict(size=5, color=y, colorscale='Viridis', opacity=0.8)))
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5, showscale=False))
fig.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))
fig.show()

# Применение PCA
pca1 = PCA(n_components=2)
pca1.fit(X)
X_pca_2d = pca1.transform(X)

# DataFrame для графика
df_pca = pd.DataFrame(data=X_pca_2d, columns=["PC1", "PC2"])
df_pca["class"] = filtered_data['species'].values

# Реализация метода k-средних
class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(42)
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids
        return self

# Кластеризация
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pca_2d)

# График кластеризации
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='X', s=200, c='red', label='Центроиды')
plt.title('K-means кластеризация (PCA проекция)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
