import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans


def train_kmeans_model(scaled_path, model_dir="models", model_name="kmeans_rfm_k4.pkl", k=4):
    print(f"Loading scaled RFM data from: {scaled_path}")
    df_scaled = pd.read_csv(scaled_path)
    features = ["Scaled_Recency", "Scaled_Frequency", "Scaled_Monetary"]
    X = df_scaled[features]
    print(f"Training K-Means model with K={k}...")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df_scaled["cluster"] = kmeans.fit_predict(X)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(kmeans, model_path)
    print(f"Trained K-Means model saved to: {model_path}")

    print("\n--- Cluster Distribution (Scaled Data) ---")
    print(df_scaled["cluster"].value_counts().sort_index())
    return kmeans, df_scaled


def profile_clusters(unscaled_path, df_scaled):
    print(f"\nLoading unscaled RFM data from: {unscaled_path}")
    df_rfm = pd.read_csv(unscaled_path)

    # Match exact customers present in scaled data
    df_merged = pd.merge(
        df_rfm,
        df_scaled[["CustomerID", "cluster"]],
        on="CustomerID",
        how="inner",
    )

    profile = df_merged.groupby("cluster").agg(
        customer_count=("cluster", "count"),
        avg_recency=("Recency", "mean"),
        med_recency=("Recency", "median"),
        avg_frequency=("Frequency", "mean"),
        med_frequency=("Frequency", "median"),
        avg_monetary=("Monetary", "mean"),
        med_monetary=("Monetary", "median"),
    ).round(2)

    print("\n--- Cluster Profiling (Unscaled RFM) ---")
    print(profile)

    return df_merged, profile


def assign_personas_and_save(
    df_merged,
    output_csv_path=os.path.join("data", "processed", "rfm_clustered.csv"),
    report_csv_path=os.path.join("reports", "persona_summary.csv"),
):
    persona_map = {
        2: "VIP",
        1: "At-Risk",
        3: "Loyal",
        0: "Hibernating",
    }

    df_final = df_merged.copy()
    df_final["Persona"] = df_final["cluster"].map(persona_map)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"\nFinal clustered data saved to: {output_csv_path}")

    persona_summary = (
        df_final.groupby("Persona")
        .agg(
            customer_count=("Persona", "count"),
            avg_recency=("Recency", "mean"),
            med_recency=("Recency", "median"),
            avg_frequency=("Frequency", "mean"),
            med_frequency=("Frequency", "median"),
            avg_monetary=("Monetary", "mean"),
            med_monetary=("Monetary", "median"),
        )
        .round(2)
    )

    os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)
    persona_summary.to_csv(report_csv_path)
    print(f"Persona summary saved to: {report_csv_path}")

    print("\n--- Final Persona Summary ---")
    print(persona_summary)

    return df_final, persona_summary


if __name__ == "__main__":
    scaled_path = os.path.join("data", "processed", "rfm_scaled.csv")
    unscaled_path = os.path.join("data", "processed", "rfm_output.csv")

    _, df_scaled = train_kmeans_model(scaled_path, k=4)
    df_merged, _ = profile_clusters(unscaled_path, df_scaled)
    df_final, persona_summary = assign_personas_and_save(df_merged)