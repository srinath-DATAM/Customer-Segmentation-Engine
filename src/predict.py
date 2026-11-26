from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

PERSONA_MAP = {
    2: "VIP",
    1: "At-Risk",
    3: "Loyal",
    0: "Hibernating",
}

PERSONA_ACTIONS = {
    "VIP": {
        "summary": "Most recent, highest frequency, and top spenders.",
        "action": (
            "Offer early access to new launches, VIP loyalty perks, and priority"
            " retention rewards."
        ),
    },
    "Loyal": {
        "summary": (
            "Recently active with steady repeat orders and moderate spend."
        ),
        "action": (
            "Run bundle offers, cross-sell/upsell campaigns, and tiered spend"
            " rewards to lift order value."
        ),
    },
    "At-Risk": {
        "summary": (
            "Strong past frequency and spend, but noticeable recent inactivity"
            " gap (~64-81 days)."
        ),
        "action": (
            "Trigger personalized win-back reminders and time-bound incentives"
            " before churn."
        ),
    },
    "Hibernating": {
        "summary": (
            "Long inactivity (~195+ days), mostly one-time buyers with low"
            " spend."
        ),
        "action": (
            "Use low-cost automated re-engagement only; avoid heavy promotional"
            " spend."
        ),
    },
}


def _resolve_path(path_value, default_rel_parts):
    """Resolve explicit path or relative path against current working dir and script dir."""
    if path_value is not None:
        candidate = Path(path_value)
        if candidate.exists():
            return candidate
        alt = BASE_DIR / candidate
        return alt if alt.exists() else candidate

    cwd_path = Path(*default_rel_parts)
    if cwd_path.exists():
        return cwd_path
    return BASE_DIR.joinpath(*default_rel_parts)


def load_model(model_path=None):
    """Load the trained K-Means model saved via joblib."""
    resolved = _resolve_path(model_path, ("models", "kmeans_rfm_k4.pkl"))
    return joblib.load(resolved)


def load_scaler(scaler_path=None, unscaled_rfm_path=None):
    """Load saved StandardScaler via joblib, or fit on unscaled RFM as fallback."""
    resolved_scaler = _resolve_path(scaler_path, ("models", "scaler.pkl"))
    if resolved_scaler.exists():
        return joblib.load(resolved_scaler)

    resolved_rfm = _resolve_path(
        unscaled_rfm_path, ("data", "processed", "rfm_output.csv")
    )
    rfm_df = pd.read_csv(resolved_rfm)
    log_df = pd.DataFrame({
        "Log_Recency": np.log1p(rfm_df["Recency"]),
        "Log_Frequency": np.log1p(rfm_df["Frequency"]),
        "Log_Monetary": np.log1p(rfm_df["Monetary"]),
    })
    scaler = StandardScaler()
    scaler.fit(log_df[["Log_Recency", "Log_Frequency", "Log_Monetary"]])
    return scaler


def _validate_rfm_inputs(recency, frequency, monetary):
    """Validate numeric bounds and return float tuple."""
    try:
        r = float(recency)
        f = float(frequency)
        m = float(monetary)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Recency, Frequency, and Monetary must be valid numbers."
        ) from exc

    if not (np.isfinite(r) and np.isfinite(f) and np.isfinite(m)):
        raise ValueError("RFM values must be finite numbers (no NaN or Inf).")

    if r < 0 or f < 0 or m < 0:
        raise ValueError(
            "Recency, Frequency, and Monetary cannot be negative."
        )

    return r, f, m


def _handle_zero_cases(r, f, m):
    """Handle F/M zero edge cases separately while allowing R=0 (today's purchase)."""
    # Case 1: No orders and no spend -> new/non-buying user
    if f == 0 and m == 0:
        return {
            "cluster": None,
            "persona": "No Purchase History",
            "summary": (
                "No completed orders or recorded spend yet (Recency=0 is valid"
                " if purchased today, but Frequency and Monetary are both 0)."
            ),
            "action": (
                "Treat as a new/unconverted user; offer a first-order welcome"
                " nudge rather than a loyalty or churn persona."
            ),
            "is_valid_customer": False,
        }

    # Case 2: Orders exist (F > 0) but spend is 0 -> free/promo/zero-value order
    if f > 0 and m == 0:
        return {
            "cluster": None,
            "persona": "Zero-Spend Activity",
            "summary": (
                "Orders are present without any monetary spend (possible free"
                " item, full discount/voucher, or missing amount)."
            ),
            "action": (
                "Verify billing/order amount data and test paid repeat"
                " conversion before assigning a spend persona."
            ),
            "is_valid_customer": False,
        }

    # Case 3: Spend > 0 but orders = 0 -> inconsistent/anomalous RFM record
    if f == 0 and m > 0:
        return {
            "cluster": None,
            "persona": "Data Anomaly (Zero Orders)",
            "summary": (
                "Monetary spend is greater than 0 while Frequency is 0, which"
                " indicates an inconsistent transaction record."
            ),
            "action": (
                "Check upstream order aggregation logic before running"
                " segmentation."
            ),
            "is_valid_customer": False,
        }

    return None


def predict_customer(recency, frequency, monetary, model=None, scaler=None):
    """Predict cluster ID, Persona label, and recommended action for a new customer."""
    r, f, m = _validate_rfm_inputs(recency, frequency, monetary)

    edge_result = _handle_zero_cases(r, f, m)
    if edge_result is not None:
        return edge_result

    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    # Note: r == 0 is valid (0 days since purchase) and safely mapped via log1p(0) = 0
    log_features = pd.DataFrame({
        "Log_Recency": [np.log1p(r)],
        "Log_Frequency": [np.log1p(f)],
        "Log_Monetary": [np.log1p(m)],
    })

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
        "is_valid_customer": True,
    }


if __name__ == "__main__":
    print("Valid (R=0, F=2, M=1200):", predict_customer(0, 2, 1200))
    print("All zeros:", predict_customer(0, 0, 0))
    print("F>0, M=0:", predict_customer(5, 1, 0))
    print("F=0, M>0:", predict_customer(5, 0, 500))