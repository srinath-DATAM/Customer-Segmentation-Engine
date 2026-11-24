import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PERSONA_MAP = {
    2: "VIP",
    1: "At-Risk",
    3: "Loyal",
    0: "Hibernating",
}

PERSONA_ACTIONS = {
    "VIP": {
        "summary": "Most recent, highest frequency, and top spenders.",
        "action": "Offer early access to new launches, VIP loyalty perks, and priority retention rewards.",
    },
    "Loyal": {
        "summary": "Recently active with steady repeat orders and moderate spend.",
        "action": "Run bundle offers, cross-sell/upsell campaigns, and tiered spend rewards to lift order value.",
    },
    "At-Risk": {
        "summary": "Strong past frequency and spend, but noticeable recent inactivity gap (~64-81 days).",
        "action": "Trigger personalized win-back reminders and time-bound incentives before churn.",
    },
    "Hibernating": {
        "summary": "Long inactivity (~195+ days), mostly one-time buyers with low spend.",
        "action": "Use low-cost automated re-engagement only; avoid heavy promotional spend.",
    },
}


def load_model(model_path=None):
    """Load the trained K-Means model saved via joblib."""
    if model_path is None:
        model_path = os.path.join("models", "kmeans_rfm_k4.pkl")
    return joblib.load(model_path)


def load_scaler(scaler_path=None, unscaled_rfm_path=None):
    """Load saved StandardScaler via joblib, or fit on unscaled RFM as fallback."""
    if scaler_path is None:
        scaler_path = os.path.join("models", "scaler.pkl")

    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)

    if unscaled_rfm_path is None:
        unscaled_rfm_path = os.path.join("data", "processed", "rfm_output.csv")

    rfm_df = pd.read_csv(unscaled_rfm_path)
    log_df = pd.DataFrame(
        {
            "Log_Recency": np.log1p(rfm_df["Recency"]),
            "Log_Frequency": np.log1p(rfm_df["Frequency"]),
            "Log_Monetary": np.log1p(rfm_df["Monetary"]),
        }
    )
    scaler = StandardScaler()
    scaler.fit(log_df[["Log_Recency", "Log_Frequency", "Log_Monetary"]])
    return scaler


def predict_customer(recency, frequency, monetary, model=None, scaler=None):
    """Predict cluster ID, Persona label, and recommended action for a new customer."""
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    log_features = pd.DataFrame(
        {
            "Log_Recency": [np.log1p(recency)],
            "Log_Frequency": [np.log1p(frequency)],
            "Log_Monetary": [np.log1p(monetary)],
        }
    )

    scaled_array = scaler.transform(log_features)
    scaled_features = pd.DataFrame(
        scaled_array,
        columns=["Scaled_Recency", "Scaled_Frequency", "Scaled_Monetary"],
    )

    cluster_id = int(model.predict(scaled_features)[0])
    persona = PERSONA_MAP.get(cluster_id, "Unknown")
    action_info = PERSONA_ACTIONS.get(persona, {"summary": "", "action": ""})

    return {
        "cluster": cluster_id,
        "persona": persona,
        "summary": action_info["summary"],
        "action": action_info["action"],
    }


if __name__ == "__main__":
    sample = predict_customer(recency=10, frequency=8, monetary=3500)
    print("Test Prediction:", sample)