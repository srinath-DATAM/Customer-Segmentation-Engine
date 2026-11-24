import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def find_optimal_k(inputpath):
    df=pd.read_csv(inputpath)
    # Select the scaled features for clustering
    X = df[['Scaled_Recency', 'Scaled_Frequency', 'Scaled_Monetary']]
    silhouette_scores=[]
    inertia_scores=[]
    k_range=range(2,12)
    print("Calculating Inertia and Silhouette Scores for K = 2 to 10...")
    for k in k_range:
        kmeans=KMeans(n_clusters=k,n_init=10,random_state=42)#default paramter max_iter and ninit
        kmeans.fit(X)
        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X,kmeans.labels_))
        print(f"K={k} Inertia: {kmeans.inertia_:.2f} Silhouette: {silhouette_score(X,kmeans.labels_):.2f}")

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(k_range,inertia_scores,marker="o",linestyle="-")
    plt.title("elbow method graph ")
    plt.xlabel("number of cluster of k")
    plt.ylabel("inertial score for distance")

    plt.subplot(1,2,2)
    plt.plot(k_range,silhouette_scores,marker="s",linestyle="-")
    plt.title("silhouette method graph")
    plt.xlabel("number of cluster of k")
    plt.ylabel("silhouette score")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("reports",exist_ok=True)
    plot_path=os.path.join("reports","elbow_silhouette_graph.png")
    plt.savefig(plot_path)
    print(f"\nOptimization graphs saved successfully to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    input_path=os.path.join("data","processed","rfm_scaled.csv")
    find_optimal_k(input_path)

