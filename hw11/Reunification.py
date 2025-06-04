from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


digits = load_digits()
X = digits.data

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=15)
plt.title('KMeans  Subgroup results')
plt.show()