import os
import streamlit as st
import pandas as pd
import numpy as np

from src.predict import (
    load_model,
    load_scaler,
    predict_customer,
    PERSONA_ACTIONS,
)

st.set_page_config(
    page_title="Customer Segmentation Engine",
    page_icon="🎯",
    layout="wide",
)

CLUSTERED_CSV = os.path.join("data", "processed", "rfm_clustered.csv")
SUMMARY_CSV = os.path.join("reports", "persona_summary.csv")


@st.cache_data
def load_clustered_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_summary_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Persona" not in df.columns and df.columns[0] != "Persona":
        df = df.rename(columns={df.columns[0]: "Persona"})
    return df


@st.cache_resource
def get_model_and_scaler():
    model = load_model()
    scaler = load_scaler()
    return model, scaler


def main():
    st.title("🎯 Customer Segmentation & Persona Explorer")
    st.caption("RFM K-Means Clustering (K=4) • Persona Insights • Live Customer Segment Tester")

    missing_files = [p for p in [CLUSTERED_CSV, SUMMARY_CSV] if not os.path.exists(p)]
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        return

    df_clustered = load_clustered_data(CLUSTERED_CSV)
    df_summary = load_summary_data(SUMMARY_CSV)
    model, scaler = get_model_and_scaler()

    # Sidebar Persona Filter
    st.sidebar.header("🔍 Persona Filter")
    personas = ["All"] + sorted(df_clustered["Persona"].dropna().unique().tolist())
    selected_persona = st.sidebar.selectbox("Select Persona", personas)

    if selected_persona == "All":
        filtered_df = df_clustered.copy()
    else:
        filtered_df = df_clustered[df_clustered["Persona"] == selected_persona].copy()

    # Top KPI Cards
    st.subheader("📊 Segment KPI Overview")
    c1, c2, c3, c4 = st.columns(4)

    total_customers = len(filtered_df)
    avg_recency = filtered_df["Recency"].mean() if total_customers else 0.0
    avg_frequency = filtered_df["Frequency"].mean() if total_customers else 0.0
    avg_monetary = filtered_df["Monetary"].mean() if total_customers else 0.0

    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Avg Recency (days)", f"{avg_recency:.1f}")
    c3.metric("Avg Frequency", f"{avg_frequency:.2f}")
    c4.metric("Avg Monetary (₹)", f"{avg_monetary:,.2f}")

    # Selected Persona Action Box
    if selected_persona != "All" and selected_persona in PERSONA_ACTIONS:
        info = PERSONA_ACTIONS[selected_persona]
        st.info(f"**{selected_persona} Profile:** {info['summary']}\n\n**Recommended Action:** {info['action']}")

    st.divider()

    # Persona Summary Table & Filtered Sample
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("📋 Persona Summary Table")
        st.dataframe(df_summary, use_container_width=True)

    with col_right:
        st.subheader(f"👥 Customer Records ({selected_persona})")
        st.dataframe(filtered_df.head(25), use_container_width="stretch")

    st.divider()

    # Visualizations Section
    st.subheader("📈 Segment Visualizations")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Customer Count by Persona**")
        persona_counts = df_clustered["Persona"].value_counts()
        st.bar_chart(persona_counts)

    with v2:
        st.markdown("**Average Monetary Spend by Persona**")
        avg_spend = df_clustered.groupby("Persona")["Monetary"].mean().sort_values(ascending=False)
        st.bar_chart(avg_spend)

    st.markdown("**Recency vs Monetary Scatter (Sampled up to 1,500 customers)**")
    sample_scatter = df_clustered.sample(min(1500, len(df_clustered)), random_state=42)
    st.scatter_chart(
        data=sample_scatter,
        x="Recency",
        y="Monetary",
        color="Persona",
    )

    st.divider()

    # Live Customer Segment Tester
    st.subheader("🧪 Live Customer Segment Tester")
    st.write("Enter customer **Recency**, **Frequency**, and **Monetary** values to predict their segment live.")

    i1, i2, i3 = st.columns(3)
    with i1:
        recency_val = st.number_input("Recency (days since last purchase)", min_value=0.0, value=10.0, step=1.0)
    with i2:
        frequency_val = st.number_input("Frequency (number of orders)", min_value=0.0, value=8.0, step=1.0)
    with i3:
        monetary_val = st.number_input("Monetary (total spend)", min_value=0.0, value=3500.0, step=50.0)

    if st.button("Predict Segment", type="primary"):
        result = predict_customer(
            recency=recency_val,
            frequency=frequency_val,
            monetary=monetary_val,
            model=model,
            scaler=scaler,
        )
        st.success(f"**Predicted Persona:** {result['persona']} (Cluster `{result['cluster']}`)")
        st.write(f"**Segment Summary:** {result['summary']}")
        st.write(f"**Recommended Action:** {result['action']}")


if __name__ == "__main__":
    main()