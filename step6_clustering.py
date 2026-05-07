import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load data
X_train, X_test, y_train, y_test = joblib.load('models/classification_data.pkl')
X_all = np.vstack([X_train, X_test])

# ── Find Best Number of Clusters ──
print("🔄 Finding best number of clusters...")
inertia     = []
sil_scores  = []
K_range     = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_all, km.labels_))
    print(f"  k={k} → Silhouette Score: {sil_scores[-1]:.4f}")

# Best k
best_k = K_range[np.argmax(sil_scores)]
print(f"\n✅ Best number of clusters: {best_k}")

# ── Train Final KMeans ──
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_all)
labels = kmeans.labels_

# Save model
joblib.dump(kmeans, 'models/kmeans.pkl')
print("✅ KMeans model saved!")

# ── Visualize with PCA ──
print("\n🔄 Creating cluster visualization...")
pca  = PCA(n_components=2)
X_2d = pca.fit_transform(X_all)
joblib.dump(pca, 'models/pca.pkl')

plt.figure(figsize=(10, 6))
colors = ['#e74c3c','#2ecc71','#3498db','#f39c12','#9b59b6','#1abc9c','#e67e22']
for i in range(best_k):
    mask = labels == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], label=f'Field Cluster {i+1}',
                alpha=0.6, s=30)

plt.title('🌾 Field Cluster Discovery (KMeans + PCA)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('models/clusters.png')
plt.show()
print("✅ Cluster chart saved as clusters.png!")

# ── Elbow Chart ──
plt.figure(figsize=(8, 4))
plt.plot(list(K_range), inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
plt.savefig('models/elbow.png')
plt.show()
print("✅ Elbow chart saved!")