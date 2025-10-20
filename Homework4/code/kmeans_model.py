import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



class KMEANS:

    def __init__(self):
        pass


    def compute_kmeans(self, X):
        pca = PCA(n_components=2)

        X_pca = pca.fit_transform(X)

        inertias = []
        silhouette_scores = []

        for k in range(2,8):
            kmeans = KMeans(n_clusters=k, random_state=42)
            print("Performing kmeans")
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)


            plt.figure(figsize=(6, 5))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow', s=30)
            plt.title(f"K-Means Clustering (K={k})")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.show()

        plt.figure(figsize=(6,5))
        plt.plot(range(2, 8), silhouette_scores, marker='o')
        plt.title("Silhouette Score for Optimal K")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.show()

        plt.figure(figsize=(6,5))
        plt.plot(range(2, 8), inertias, marker='o')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
        plt.show()


        return inertias, silhouette_scores

    





