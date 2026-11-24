import os
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan


def evaluate_silhouette(X, labels):
    """Calculate silhouette score safely even if noise (-1) or single cluster exists in each model."""
    unique_labels = set(labels)
    non_noise_label = [x for x in unique_labels if x != -1]
    if len(non_noise_label) < 2:
        return None
    return silhouette_score(X, labels)


def run_multi_model_benchmark(input_path, k=4):
    print(f"Loading scaled RFM data from: {input_path}")
    df = pd.read_csv(input_path)
    X = df[["Scaled_Recency", "Scaled_Frequency", "Scaled_Monetary"]]
    result = []

    # 1. K-Means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_label = kmeans.fit_predict(X)
    km_score = evaluate_silhouette(X, kmeans_label)
    result.append({
        "model": f"K-Means (k={k})",
        "cluster_found": len(set(kmeans_label)),
        "silhouette_score": round(km_score, 4) if km_score is not None else None,
        "Noise_count": int((kmeans_label == -1).sum())
    })

    # 2. Agglomerative Hierarchical
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agg.fit_predict(X)
    agg_score = evaluate_silhouette(X, agg_labels)
    result.append({
        "model": f"Agglomerative Clustering (k={k})",
        "cluster_found": len(set(agg_labels)),
        "silhouette_score": round(agg_score, 4) if agg_score is not None else None,
        "Noise_count": int((agg_labels == -1).sum())
    })

    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)
    dbscan_score = evaluate_silhouette(X, dbscan_labels)
    result.append({
        "model": "DBSCAN (eps=0.5)",
        "cluster_found": len([c for c in set(dbscan_labels) if c != -1]),
        "silhouette_score": round(dbscan_score, 4) if dbscan_score is not None else None,
        "Noise_count": int((dbscan_labels == -1).sum())
    })

    # 4. HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
    hdb_labels = clusterer.fit_predict(X)
    hdb_score = evaluate_silhouette(X, hdb_labels)
    result.append({
        "model": "HDBSCAN (min_cluster_size=30)",
        "cluster_found": len([c for c in set(hdb_labels) if c != -1]),
        "silhouette_score": round(hdb_score, 4) if hdb_score is not None else None,
        "Noise_count": int((hdb_labels == -1).sum())
    })

    # 5. Gaussian Mixture Model (GMM)
    # 5. Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
    gmm_label = gmm.fit_predict(X)
    gmm_score = evaluate_silhouette(X, gmm_label)
    result.append({
        "model": f"GMM (k={k})",
        "cluster_found": len(set(gmm_label)),
        "silhouette_score": round(gmm_score, 4) if gmm_score is not None else None,
        "Noise_count": int((gmm_label == -1).sum())
    })

    summary_df = pd.DataFrame(result)
    print("\n--- Multi-Model Benchmarking Summary ---")
    print(summary_df.to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "model_benchmark_summary.csv")
    summary_df.to_csv(report_path, index=False)
    print(f"\nBenchmark summary saved to: {report_path}")


if __name__ == "__main__":
    input_path = os.path.join("data", "processed", "rfm_scaled.csv")
    run_multi_model_benchmark(input_path, k=4)