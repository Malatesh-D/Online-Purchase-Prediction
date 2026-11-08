"""
Online Shopping Purchase Intention Prediction - Interactive Streamlit Dashboard.
Academic Field Survey Edition (N=1,200 Manual Survey Responses).
Provides real-time machine learning inference, interactive survey data exploration,
and downloadable CSV & Excel survey assets.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALL_FEATURE_COLUMNS,
    BEST_MODEL_PATH,
    CUSTOMER_TYPE_OPTIONS,
    DATA_EXCEL_PATH,
    DATA_FILE_PATH,
    DAY_OPTIONS,
    DEVICE_OPTIONS,
    FEATURE_DESCRIPTIONS,
    GENDER_OPTIONS,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MONTH_OPTIONS,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    REFERRAL_OPTIONS,
    TARGET_COLUMN,
)
from src.data_loader import inspect_data, load_raw_data
from src.predict import PurchasePredictor
from src.utils import load_json

# ---------------------------------------------------------
# Page Configuration & Pure White Aesthetic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ShopSense AI | Academic Survey Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Pure White Theme Override */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    
    /* Global Crisp Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p, span, label, div {
        color: #334155;
    }

    /* Main container padding */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    
    /* Modern Attractive KPI Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.80rem;
        color: #2563EB;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 100%);
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1E3A8A !important;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #475569 !important;
        line-height: 1.5;
        margin-bottom: 0px;
    }
    .academic-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Prediction Result Cards */
    .result-card-success {
        background: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(22, 101, 52, 0.06);
    }
    .result-card-failure {
        background: #FEF2F2;
        border: 2px solid #FECACA;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(153, 27, 27, 0.06);
    }
    .status-badge-success {
        background: #16A34A;
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-failure {
        background: #DC2626;
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Section Cards */
    .section-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Streamlit widgets styling for white theme */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    .stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        border: 1px solid #2563EB !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton > button:hover {
        background-color: #EFF6FF !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Caching Functions
# ---------------------------------------------------------
@st.cache_data
def get_dataset():
    """Load and cache the manual survey dataset."""
    try:
        return load_raw_data()
    except Exception as e:
        return None


@st.cache_data
def get_model_comparison_data():
    """Load model evaluation comparison table."""
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None


@st.cache_resource
def load_predictor():
    """Instantiate and cache the predictor."""
    try:
        if BEST_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
            return PurchasePredictor()
    except Exception as e:
        st.error(f"Error loading prediction pipeline: {e}")
    return None


# ---------------------------------------------------------
# Sidebar Navigation (Strictly 3 Pages)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 1.8rem;">🛍️</span>
            <div>
                <span style="font-size: 1.25rem; font-weight: 800; color: #1E3A8A; display: block; line-height: 1.1;">ShopSense AI</span>
                <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Academic Survey Edition</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("E-Commerce Purchase Intention Predictive Analytics")
    st.divider()

    navigation = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "📊 Data Explorer",
            "🔮 Purchase Prediction",
        ],
        index=0,
    )

    st.divider()

    # System status indicator
    model_exists = BEST_MODEL_PATH.exists()
    data_exists = DATA_FILE_PATH.exists()

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;'>Project Datasets</div>", unsafe_allow_html=True)
    if data_exists:
        st.success("✅ Survey Data Loaded (1,200 Rows)")
        with open(DATA_FILE_PATH, "rb") as f:
            st.download_button(
                label="📥 Download Survey CSV",
                data=f,
                file_name="manual_shopper_survey_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if DATA_EXCEL_PATH.exists():
            with open(DATA_EXCEL_PATH, "rb") as f_xl:
                st.download_button(
                    label="📊 Download Excel Workbook",
                    data=f_xl,
                    file_name="manual_shopper_survey_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    else:
        st.error("❌ Survey data not found")

    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #0F172A; margin-top: 14px; margin-bottom: 6px;'>Trained Model</div>", unsafe_allow_html=True)
    if model_exists:
        st.info("🧠 Best Model: **Logistic Regression** (96.0% ROC-AUC)")
    else:
        st.warning("⚠️ Model training required")

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8; text-align: center;">
            Academic Predictive Analytics Project<br>
            Validated Field Survey Dataset & ML Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load data and artifacts
raw_df = get_dataset()
comparison_df = get_model_comparison_data()
metadata = load_json(MODEL_METADATA_PATH) if MODEL_METADATA_PATH.exists() else {}


# =========================================================
# PAGE 1: HOME DASHBOARD
# =========================================================
if navigation == "🏠 Home Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <span class="academic-badge">🎓 Academic Predictive Analytics Project</span>
            <div class="hero-title">Predict Whether a Person Will Make an Online Purchase</div>
            <div class="hero-subtitle">
                An end-to-end Machine Learning system trained on <b>1,200 manually collected shopper survey responses</b>. 
                Features demographic profiling, acquisition attribution, session behavioral tracking, and real-time conversion prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if raw_df is not None:
        total_sessions = len(raw_df)
        total_buyers = int((raw_df[TARGET_COLUMN] == "Yes").sum())
        conversion_rate = (total_buyers / total_sessions) * 100
        best_model_name = metadata.get("best_model_name", "Logistic Regression")
        best_roc = metadata.get("metrics", {}).get("Test ROC-AUC", 0.9602)
        best_acc = metadata.get("metrics", {}).get("Test Accuracy", 0.8625)

        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Survey Respondents</div>
                    <div class="metric-value">{total_sessions:,}</div>
                    <div class="metric-delta">👥 Field Survey Sample</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Completed Purchases</div>
                    <div class="metric-value">{total_buyers:,}</div>
                    <div class="metric-delta">🎯 {conversion_rate:.1f}% Conversion Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Best Model</div>
                    <div class="metric-value" style="font-size: 1.45rem;">{best_model_name}</div>
                    <div class="metric-delta">⚡ ROC-AUC: {best_roc:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Test Accuracy</div>
                    <div class="metric-value">{best_acc:.1%}</div>
                    <div class="metric-delta">📊 5 Models Evaluated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📱 Conversion Rate by Primary Device")
            device_conv = (
                raw_df.groupby("Primary_Device")[TARGET_COLUMN]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .reset_index(name="Conversion_Rate_Pct")
            )
            fig_device = px.bar(
                device_conv,
                x="Primary_Device",
                y="Conversion_Rate_Pct",
                color="Primary_Device",
                color_discrete_sequence=["#2563EB", "#0D9488", "#F59E0B"],
                text="Conversion_Rate_Pct",
            )
            fig_device.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker_line_color="#0F172A",
                marker_line_width=1,
            )
            fig_device.update_layout(
                yaxis_title="Conversion Rate (%)",
                xaxis_title="Shopper Device",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_device, use_container_width=True)
            st.caption("💡 **Key Finding:** Laptop/Desktop users demonstrate the highest conversion likelihood due to deeper browsing engagement.")

        with col_right:
            st.subheader("📣 Conversion by Referral Channel")
            channel_conv = (
                raw_df.groupby("Referral_Channel")[TARGET_COLUMN]
                .apply(lambda s: (s == "Yes").mean() * 100)
                .reset_index(name="Conversion_Rate_Pct")
                .sort_values(by="Conversion_Rate_Pct", ascending=False)
            )
            fig_channel = px.bar(
                channel_conv,
                x="Referral_Channel",
                y="Conversion_Rate_Pct",
                color="Referral_Channel",
                color_discrete_sequence=["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
                text="Conversion_Rate_Pct",
            )
            fig_channel.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                marker_line_color="#0F172A",
                marker_line_width=1,
            )
            fig_channel.update_layout(
                yaxis_title="Conversion Rate (%)",
                xaxis_title="Referral Channel",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_channel, use_container_width=True)
            st.caption("💡 **Key Finding:** Direct visits and organic Google searches yield top conversion rates over generic social ads.")

        # Survey Methodology Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Manual Survey Methodology & Academic Overview")
        with st.expander("📖 View Survey Collection Architecture & Codebook Details", expanded=True):
            st.markdown(
                """
                - **Survey Population:** $N = 1,200$ verified online shoppers across demographic age groups (18–65 years).
                - **Data Acquisition Mode:** Administered questionnaire capturing actual browsing sessions, cart engagement, and completed purchases.
                - **Dual Output Variables:**
                  - `Completed_Purchase`: Binary classification target (`Yes` / `No`).
                  - `Order_Amount_Spent_USD`: Continuous order value recorded for converting sessions ($0.00 for non-purchasing visitors).
                - **Model Performance Benchmark:**
                  - **Logistic Regression:** 86.3% Accuracy, 96.0% ROC-AUC, 83.7% Tuned F1-Score (Best Selected Model).
                  - **Gradient Boosting:** 88.8% Accuracy, 95.3% ROC-AUC, 82.7% Tuned F1-Score.
                  - **XGBoost Classifier:** 85.8% Accuracy, 95.1% ROC-AUC, 81.9% Tuned F1-Score.
                  - **Random Forest:** 81.3% Accuracy, 93.5% ROC-AUC, 81.1% Tuned F1-Score.
                """
            )


# =========================================================
# PAGE 2: DATA EXPLORER
# =========================================================
elif navigation == "📊 Data Explorer":
    st.markdown(
        """
        <div class="hero-banner">
            <span class="academic-badge">📊 Data Inspection & Distribution Analysis</span>
            <div class="hero-title">Shopper Survey Dataset Explorer</div>
            <div class="hero-subtitle">
                Explore the raw survey responses, inspect demographic distributions, filter records by visitor attributes, and download the dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if raw_df is not None:
        # Filters row
        st.markdown("#### 🔍 Filter Survey Responses")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            device_filter = st.multiselect(
                "Primary Device",
                options=DEVICE_OPTIONS,
                default=DEVICE_OPTIONS,
            )

        with f_col2:
            customer_filter = st.multiselect(
                "Customer Type",
                options=CUSTOMER_TYPE_OPTIONS,
                default=CUSTOMER_TYPE_OPTIONS,
            )

        with f_col3:
            channel_filter = st.multiselect(
                "Referral Channel",
                options=REFERRAL_OPTIONS,
                default=REFERRAL_OPTIONS,
            )

        with f_col4:
            purchase_filter = st.multiselect(
                "Completed Purchase",
                options=["Yes", "No"],
                default=["Yes", "No"],
            )

        # Apply filters
        filtered_df = raw_df[
            (raw_df["Primary_Device"].isin(device_filter))
            & (raw_df["Customer_Type"].isin(customer_filter))
            & (raw_df["Referral_Channel"].isin(channel_filter))
            & (raw_df[TARGET_COLUMN].isin(purchase_filter))
        ]

        st.markdown(f"**Showing {len(filtered_df):,} of {len(raw_df):,} survey records**")
        st.dataframe(filtered_df, use_container_width=True, height=360)

        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name="filtered_shopper_survey_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d_col2:
            if DATA_EXCEL_PATH.exists():
                with open(DATA_EXCEL_PATH, "rb") as f_xl:
                    st.download_button(
                        label="📊 Download Complete Excel Workbook (.xlsx)",
                        data=f_xl,
                        file_name="manual_shopper_survey_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Statistical Relationships & Visual Distributions")

        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
            "Cart Value vs. Drop-off Exit Rate",
            "Age & Gender Demographics",
            "Browsing Time vs. Products Viewed",
        ])

        with chart_tab1:
            fig_scatter = px.scatter(
                filtered_df,
                x="Cart_Page_Value_Score",
                y="Drop_Off_Exit_Rate_Pct",
                color=TARGET_COLUMN,
                color_discrete_map={"Yes": "#16A34A", "No": "#DC2626"},
                size="Products_Viewed",
                hover_data=["Respondent_ID", "Age", "Primary_Device", "Order_Amount_Spent_USD"],
                labels={
                    "Cart_Page_Value_Score": "Cart Page Value Score ($)",
                    "Drop_Off_Exit_Rate_Pct": "Drop-Off Exit Rate (%)",
                    TARGET_COLUMN: "Completed Purchase",
                },
            )
            fig_scatter.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Insight: Purchasers cluster predominantly in the high Cart Page Value region (> $30) and low drop-off exit rate (< 4.5%).")

        with chart_tab2:
            fig_age = px.histogram(
                filtered_df,
                x="Age",
                color=TARGET_COLUMN,
                barmode="overlay",
                nbins=25,
                color_discrete_map={"Yes": "#16A34A", "No": "#94A3B8"},
                labels={"Age": "Shopper Age", TARGET_COLUMN: "Completed Purchase"},
            )
            fig_age.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=380,
            )
            st.plotly_chart(fig_age, use_container_width=True)
            st.caption("Insight: Young adult shoppers (20–30 years) constitute the primary respondents with high purchasing frequency.")

        with chart_tab3:
            fig_time = px.scatter(
                filtered_df,
                x="Browsing_Time_Minutes",
                y="Products_Viewed",
                color=TARGET_COLUMN,
                color_discrete_map={"Yes": "#16A34A", "No": "#DC2626"},
                labels={
                    "Browsing_Time_Minutes": "Browsing Time (Minutes)",
                    "Products_Viewed": "Products Viewed",
                    TARGET_COLUMN: "Completed Purchase",
                },
            )
            fig_time.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=380,
            )
            st.plotly_chart(fig_time, use_container_width=True)


# =========================================================
# PAGE 3: PURCHASE PREDICTION
# =========================================================
elif navigation == "🔮 Purchase Prediction":
    st.markdown(
        """
        <div class="hero-banner">
            <span class="academic-badge">🔮 Real-Time Machine Learning Inference</span>
            <div class="hero-title">Predict Shopper Purchase Intention</div>
            <div class="hero-subtitle">
                Fill in the respondent's survey answers or shopping visit attributes. 
                The trained model evaluates conversion likelihood, predicts completed purchase class, and provides academic and commercial recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    if predictor is None:
        st.error("⚠️ Prediction pipeline artifacts are missing. Run `python src/train.py` to train and generate the model.")
    else:
        # Quick Presets
        st.markdown("#### ⚡ Quick Example Profiles")
        preset_cols = st.columns(3)

        preset_data = None
        with preset_cols[0]:
            if st.button("🛒 High-Intent Mobile Buyer", use_container_width=True):
                st.session_state["p_age"] = 23
                st.session_state["p_gender"] = "Male"
                st.session_state["p_customer"] = "Returning Customer"
                st.session_state["p_device"] = "Laptop / Desktop"
                st.session_state["p_month"] = "November"
                st.session_state["p_day"] = "Weekday"
                st.session_state["p_channel"] = "Direct Website Visit"
                st.session_state["p_products"] = 16
                st.session_state["p_time"] = 17.7
                st.session_state["p_checkout"] = 2
                st.session_state["p_cart_val"] = 80.70
                st.session_state["p_exit_rate"] = 3.10

        with preset_cols[1]:
            if st.button("👀 Window Shopper (Non-Buyer)", use_container_width=True):
                st.session_state["p_age"] = 41
                st.session_state["p_gender"] = "Female"
                st.session_state["p_customer"] = "Returning Customer"
                st.session_state["p_device"] = "Smartphone"
                st.session_state["p_month"] = "February"
                st.session_state["p_day"] = "Weekday"
                st.session_state["p_channel"] = "Google Search"
                st.session_state["p_products"] = 2
                st.session_state["p_time"] = 5.8
                st.session_state["p_checkout"] = 0
                st.session_state["p_cart_val"] = 0.0
                st.session_state["p_exit_rate"] = 5.44

        with preset_cols[2]:
            if st.button("🎁 Holiday Social Shopper", use_container_width=True):
                st.session_state["p_age"] = 28
                st.session_state["p_gender"] = "Female"
                st.session_state["p_customer"] = "New Visitor"
                st.session_state["p_device"] = "Smartphone"
                st.session_state["p_month"] = "December"
                st.session_state["p_day"] = "Weekend"
                st.session_state["p_channel"] = "Instagram / Social Ad"
                st.session_state["p_products"] = 14
                st.session_state["p_time"] = 16.5
                st.session_state["p_checkout"] = 1
                st.session_state["p_cart_val"] = 95.50
                st.session_state["p_exit_rate"] = 1.45

        st.markdown("<br>", unsafe_allow_html=True)

        # Input Form
        with st.form("prediction_form"):
            st.markdown("### 📋 Shopper Profile & Survey Input Data")

            # Section 1: Demographics & Device
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;'>1. Shopper Demographics & Device</div>", unsafe_allow_html=True)
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)

            with d_col1:
                age_val = st.slider(
                    "Shopper Age (Years)",
                    min_value=18,
                    max_value=70,
                    value=st.session_state.get("p_age", 24),
                    help=FEATURE_DESCRIPTIONS["Age"],
                )

            with d_col2:
                gender_val = st.selectbox(
                    "Gender Identity",
                    options=GENDER_OPTIONS,
                    index=GENDER_OPTIONS.index(st.session_state.get("p_gender", "Female")),
                    help=FEATURE_DESCRIPTIONS["Gender"],
                )

            with d_col3:
                customer_val = st.selectbox(
                    "Customer Type",
                    options=CUSTOMER_TYPE_OPTIONS,
                    index=CUSTOMER_TYPE_OPTIONS.index(st.session_state.get("p_customer", "Returning Customer")),
                    help=FEATURE_DESCRIPTIONS["Customer_Type"],
                )

            with d_col4:
                device_val = st.selectbox(
                    "Primary Device Used",
                    options=DEVICE_OPTIONS,
                    index=DEVICE_OPTIONS.index(st.session_state.get("p_device", "Smartphone")),
                    help=FEATURE_DESCRIPTIONS["Primary_Device"],
                )

            st.divider()

            # Section 2: Visit Timing & Referral
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;'>2. Visit Timing & Acquisition Channel</div>", unsafe_allow_html=True)
            t_col1, t_col2, t_col3 = st.columns(3)

            with t_col1:
                month_val = st.selectbox(
                    "Shopping Month",
                    options=MONTH_OPTIONS,
                    index=MONTH_OPTIONS.index(st.session_state.get("p_month", "November")),
                    help=FEATURE_DESCRIPTIONS["Shopping_Month"],
                )

            with t_col2:
                day_val = st.selectbox(
                    "Shopping Day",
                    options=DAY_OPTIONS,
                    index=DAY_OPTIONS.index(st.session_state.get("p_day", "Weekday")),
                    help=FEATURE_DESCRIPTIONS["Shopping_Day"],
                )

            with t_col3:
                channel_val = st.selectbox(
                    "Referral Acquisition Channel",
                    options=REFERRAL_OPTIONS,
                    index=REFERRAL_OPTIONS.index(st.session_state.get("p_channel", "Instagram / Social Ad")),
                    help=FEATURE_DESCRIPTIONS["Referral_Channel"],
                )

            st.divider()

            # Section 3: Browsing & Cart Behavior
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px;'>3. Session Browsing & Cart Engagement</div>", unsafe_allow_html=True)
            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

            with b_col1:
                products_val = st.number_input(
                    "Products Viewed",
                    min_value=1,
                    max_value=150,
                    value=st.session_state.get("p_products", 15),
                    step=1,
                    help=FEATURE_DESCRIPTIONS["Products_Viewed"],
                )

            with b_col2:
                time_val = st.number_input(
                    "Browsing Time (Mins)",
                    min_value=0.5,
                    max_value=180.0,
                    value=float(st.session_state.get("p_time", 18.5)),
                    step=0.5,
                    help=FEATURE_DESCRIPTIONS["Browsing_Time_Minutes"],
                )

            with b_col3:
                checkout_val = st.number_input(
                    "Cart/Checkout Pages",
                    min_value=0,
                    max_value=15,
                    value=int(st.session_state.get("p_checkout", 2)),
                    step=1,
                    help=FEATURE_DESCRIPTIONS["Cart_Checkout_Pages_Visited"],
                )

            with b_col4:
                cart_val = st.number_input(
                    "Cart Page Value ($)",
                    min_value=0.0,
                    max_value=300.0,
                    value=float(st.session_state.get("p_cart_val", 65.0)),
                    step=5.0,
                    help=FEATURE_DESCRIPTIONS["Cart_Page_Value_Score"],
                )

            with b_col5:
                exit_rate_val = st.slider(
                    "Exit Rate (%)",
                    min_value=0.1,
                    max_value=15.0,
                    value=float(st.session_state.get("p_exit_rate", 2.5)),
                    step=0.1,
                    help=FEATURE_DESCRIPTIONS["Drop_Off_Exit_Rate_Pct"],
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(
                "🔮 Predict Purchase Intention",
                use_container_width=True,
            )

        if submit_button:
            # Package input data
            input_dict = {
                "Age": age_val,
                "Gender": gender_val,
                "Customer_Type": customer_val,
                "Primary_Device": device_val,
                "Shopping_Month": month_val,
                "Shopping_Day": day_val,
                "Referral_Channel": channel_val,
                "Products_Viewed": products_val,
                "Browsing_Time_Minutes": time_val,
                "Cart_Checkout_Pages_Visited": checkout_val,
                "Cart_Page_Value_Score": cart_val,
                "Drop_Off_Exit_Rate_Pct": exit_rate_val,
            }

            try:
                result = predictor.predict(input_dict)
                is_purchase = (result["raw_class"] == 1)

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("🎯 Prediction Results & Analytics")

                if is_purchase:
                    st.markdown(
                        f"""
                        <div class="result-card-success">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span class="status-badge-success">✅ COMPLETED PURCHASE (LIKELY BUYER)</span>
                                <span style="font-size: 0.88rem; font-weight: 700; color: #166534;">Intent Category: {result['risk_level']}</span>
                            </div>
                            <div style="font-size: 2.1rem; font-weight: 800; color: #14532D; margin-bottom: 6px;">
                                {result['purchase_probability_pct']} Purchase Probability
                            </div>
                            <p style="font-size: 0.95rem; color: #166534; line-height: 1.5; margin-bottom: 12px;">
                                {result['interpretation']}
                            </p>
                            <div style="display: flex; gap: 24px; border-top: 1px solid #BBF7D0; padding-top: 12px; margin-top: 10px;">
                                <div>
                                    <div style="font-size: 0.78rem; color: #166534; font-weight: 600;">CONFIDENCE SCORE</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #14532D;">{result['confidence_score_pct']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.78rem; color: #166534; font-weight: 600;">ESTIMATED ORDER VALUE</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #14532D;">{result['estimated_order_amount_usd']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.78rem; color: #166534; font-weight: 600;">DECISION THRESHOLD</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #14532D;">{result['threshold_used']:.2f}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-failure">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span class="status-badge-failure">❌ NO PURCHASE (UNLIKELY TO BUY)</span>
                                <span style="font-size: 0.88rem; font-weight: 700; color: #991B1B;">Intent Category: {result['risk_level']}</span>
                            </div>
                            <div style="font-size: 2.1rem; font-weight: 800; color: #7F1D1D; margin-bottom: 6px;">
                                {result['non_purchase_probability_pct']} Non-Purchase Probability
                            </div>
                            <p style="font-size: 0.95rem; color: #991B1B; line-height: 1.5; margin-bottom: 12px;">
                                {result['interpretation']}
                            </p>
                            <div style="display: flex; gap: 24px; border-top: 1px solid #FECACA; padding-top: 12px; margin-top: 10px;">
                                <div>
                                    <div style="font-size: 0.78rem; color: #991B1B; font-weight: 600;">CONFIDENCE SCORE</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #7F1D1D;">{result['confidence_score_pct']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.78rem; color: #991B1B; font-weight: 600;">PURCHASE PROBABILITY</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #7F1D1D;">{result['purchase_probability_pct']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.78rem; color: #991B1B; font-weight: 600;">DECISION THRESHOLD</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #7F1D1D;">{result['threshold_used']:.2f}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Visual probability meter
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📊 Purchase Probability Breakdown")
                prob_pos_pct = result["purchase_probability"] * 100
                prob_neg_pct = result["non_purchase_probability"] * 100

                fig_gauge = go.Figure()
                fig_gauge.add_trace(
                    go.Bar(
                        y=["Intent"],
                        x=[prob_pos_pct],
                        name="Purchase Probability",
                        orientation="h",
                        marker=dict(color="#16A34A"),
                        text=f"{prob_pos_pct:.1f}%",
                        textposition="inside",
                    )
                )
                fig_gauge.add_trace(
                    go.Bar(
                        y=["Intent"],
                        x=[prob_neg_pct],
                        name="Non-Purchase Probability",
                        orientation="h",
                        marker=dict(color="#E2E8F0"),
                        text=f"{prob_neg_pct:.1f}%",
                        textposition="inside",
                    )
                )
                fig_gauge.update_layout(
                    barmode="stack",
                    height=100,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                    yaxis=dict(showticklabels=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Inference error encountered: {e}")
