import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import base64

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AstraZeneca Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).parent

LR_MODEL_PATH = BASE_DIR / "linear_regression_model.joblib"
RF_MODEL_PATH = BASE_DIR / "random_forest_model.joblib"

LOGO_PATH = BASE_DIR / "Astrazeneca_Logo.png"
BUILDING_IMAGE = BASE_DIR / "AstraZeneca_England.png"

# =========================================================
# MODEL METRICS
# =========================================================
LR_RMSE = 0.8245
RF_RMSE = 1.6219

LR_R2 = 0.8970
RF_R2 = 0.1777

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    try:
        lr_model = joblib.load(LR_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)

        return lr_model, rf_model

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

lr_model, rf_model = load_models()

# =========================================================
# BASE64 IMAGE
# =========================================================
def get_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_image = get_base64(BUILDING_IMAGE)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    f"""
    <style>

    /* APP BACKGROUND */
    .stApp {{
        background-color: #ECECEC;
    }}

    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #121826 0%, #1E2235 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
        color: white !important;
    }}

    /* SIDEBAR TEXT FIX */
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}

    /* MAIN CONTAINER */
    .main-container {{
        background: rgba(255,255,255,0.94);
        border-radius: 24px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
        color: #111 !important;
    }}

    /* GLOBAL TEXT */
    label, p, div, span, h1, h2, h3, h4 {{
        color: #111;
    }}

    /* TITLE */
    .main-title {{
        color: #6A0DAD !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }}

    .subtitle {{
        color: #555 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }}

    /* ============================
       SELECTBOX FIX (IMPORTANT)
       ============================ */

    /* selected box */
    div[data-baseweb="select"] > div {{
        background-color: #1D2233 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    /* selected text */
    div[data-baseweb="select"] span {{
        color: white !important;
    }}

    /* dropdown menu */
    div[role="listbox"] {{
        background-color: #1D2233 !important;
        color: white !important;
    }}

    /* dropdown options */
    div[role="option"] {{
        background-color: #1D2233 !important;
        color: white !important;
    }}

    div[role="option"]:hover {{
        background-color: #2A3150 !important;
    }}

    /* RADIO BUTTONS */
    .stRadio > div {{
        background-color: transparent !important;
    }}

    .stRadio label {{
        color: white !important;   /* FIXED for sidebar + dark bg */
        font-weight: 500;
    }}

    /* BUTTON */
    .stButton > button {{
        background: #6A0DAD;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-weight: bold;
        transition: 0.3s;
    }}

    .stButton > button:hover {{
        background: #4E0B80;
        color: white !important;
    }}

    /* METRIC CARDS */
    .metric-card {{
        background: white;
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
        text-align: center;
    }}

    .metric-value {{
        color: #16A34A !important;
        font-size: 2rem;
        font-weight: 800;
    }}

    .metric-title {{
        color: #666 !important;
        font-size: 0.9rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# ATC INFORMATION
# =========================================================
ATC_INFO = {
    "M01AB": {
        "name": "Anti-inflammatory and antirheumatic products",
        "description": "Used for arthritis, pain and inflammation.",
        "drug": "Vimovo"
    },

    "M01AE": {
        "name": "Propionic acid derivatives",
        "description": "NSAIDs used for pain management.",
        "drug": "Ibuprofen therapies"
    },

    "R03": {
        "name": "Drugs for obstructive airway diseases",
        "description": "Used for asthma and COPD.",
        "drug": "Symbicort"
    }
}

ATC_COLUMNS = sorted(list(ATC_INFO.keys()))

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.image(str(LOGO_PATH), width=150)

    st.markdown(
        "<div class='sidebar-title'>ATC Classification</div>",
        unsafe_allow_html=True
    )

    sidebar_atc = st.selectbox(
        "Choose an ATC Code",
        ATC_COLUMNS
    )

    st.markdown(f"### {sidebar_atc}")

    st.markdown(
        f"<div class='sidebar-sub'>{ATC_INFO[sidebar_atc]['name']}</div>",
        unsafe_allow_html=True
    )

    st.markdown("### Description")
    st.write(ATC_INFO[sidebar_atc]["description"])

    st.markdown("### AstraZeneca Example")
    st.info(ATC_INFO[sidebar_atc]["drug"])

# =========================================================
# MAIN CONTENT
# =========================================================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>AstraZeneca Global Demand Forecasting</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-powered pharmaceutical demand forecasting using Machine Learning.</div>",
    unsafe_allow_html=True
)

# =========================================================
# INPUT SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:

    atc_classification = st.selectbox(
        "ATC Classification",
        ATC_COLUMNS
    )

with col2:

    current_year = datetime.now().year

    selected_year = st.selectbox(
        "Select Year",
        list(range(current_year, current_year + 6))
    )

# =========================================================
# FORECAST MODE
# =========================================================
forecast_mode = st.radio(
    "Forecast Mode",
    ["Yearly Forecast", "Monthly Forecast"],
    horizontal=True
)

selected_month = None

if forecast_mode == "Monthly Forecast":

    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    selected_month_name = st.selectbox(
        "Select Month",
        list(month_map.keys())
    )

    selected_month = month_map[selected_month_name]

# =========================================================
# MODEL CHOICE
# =========================================================
st.markdown("## Model Selection")

model_choice = st.radio(
    "",
    [
        f"Linear Regression | RMSE: {LR_RMSE} | R²: {LR_R2}",
        f"Random Forest | RMSE: {RF_RMSE} | R²: {RF_R2}"
    ]
)

selected_model = (
    lr_model if "Linear Regression" in model_choice else rf_model
)

# =========================================================
# IMAGE + BUTTON
# =========================================================
left_col, right_col = st.columns([1.1, 1])

with left_col:

    generate = st.button("Generate Forecast")

with right_col:

    st.image(str(BUILDING_IMAGE), use_container_width=True)

# =========================================================
# PREDICTION
# =========================================================
if generate:

    # =====================================================
    # DATE RANGE
    # =====================================================
    if forecast_mode == "Yearly Forecast":

        start_date = datetime(selected_year, 1, 1)
        end_date = datetime(selected_year, 12, 31)

    else:

        start_date = datetime(selected_year, selected_month, 1)

        if selected_month == 12:

            end_date = datetime(selected_year, 12, 31)

        else:

            end_date = datetime(
                selected_year,
                selected_month + 1,
                1
            ) - timedelta(days=1)

    # =====================================================
    # FUTURE DATES
    # =====================================================
    future_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    future_df = pd.DataFrame(index=future_dates)

    future_df["Year"] = future_df.index.year
    future_df["Month"] = future_df.index.month
    future_df["DayOfWeek"] = future_df.index.dayofweek

    # =====================================================
    # EXACT FEATURES MODEL EXPECTS
    # =====================================================
    TRAIN_FEATURES = [
        "Year",
        "Month",
        "DayOfWeek",

        "M01AB_Lag1",
        "M01AE_Lag1",
        "R03_Lag1",

        "M01AB_Lag2",
        "M01AE_Lag2",
        "R03_Lag2",

        "M01AB_Lag7",
        "M01AE_Lag7",
        "R03_Lag7",

        "M01AB_RollingMean7",
        "M01AE_RollingMean7",
        "R03_RollingMean7",
    ]

    # =====================================================
    # CREATE FEATURES
    # =====================================================
    for feature in TRAIN_FEATURES:

        if feature not in ["Year", "Month", "DayOfWeek"]:

            future_df[feature] = 350

    # Rolling means
    future_df["M01AB_RollingMean7"] = 410
    future_df["M01AE_RollingMean7"] = 410
    future_df["R03_RollingMean7"] = 410

    # =====================================================
    # FINAL INPUT
    # =====================================================
    X_future = future_df[TRAIN_FEATURES]

    # =====================================================
    # PREDICTIONS
    # =====================================================
    predictions = selected_model.predict(X_future)

    target_index = ATC_COLUMNS.index(atc_classification)

    predicted_daily = predictions[:, target_index]

    total_prediction = int(np.sum(predicted_daily))

    # =====================================================
    # RESULTS
    # =====================================================
    st.markdown("---")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ATC Classification</div>
                <div class="metric-value">{atc_classification}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Forecast Year</div>
                <div class="metric-value">{selected_year}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Demand</div>
                <div class="metric-value">{total_prediction:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # CHART
    # =====================================================
    chart_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Demand": predicted_daily
    })

    chart_df.set_index("Date", inplace=True)

    st.markdown("## Forecast Trend")

    st.line_chart(chart_df)

    # =====================================================
    # TABLE
    # =====================================================
    st.markdown("## Forecast Data")

    st.dataframe(
        chart_df.head(30),
        use_container_width=True
    )

st.markdown("</div>", unsafe_allow_html=True)
