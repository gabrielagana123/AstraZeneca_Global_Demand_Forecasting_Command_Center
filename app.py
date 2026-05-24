import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import base64 # Import base64 for image embedding

# Model filenames and their associated RMSE and R^2 for display
LR_MODEL_FILENAME = 'az_lr_macro_model.pkl'
RF_MODEL_FILENAME = 'az_rf_micro_model.pkl'

# These values are now directly embedded from the notebook's execution results
LR_RMSE = 0.8245
RF_RMSE = 1.6219
LR_R2 = 0.8970
RF_R2 = 0.1777

# Load the trained models
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load(LR_MODEL_FILENAME)
        rf_model = joblib.load(RF_MODEL_FILENAME)
        return lr_model, rf_model
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}. Please ensure the model training pipeline was run and models were saved.")
        st.stop()

lr_model, rf_model = load_models()

# Function to convert image to base64
def img_to_base64(filepath):
    with open(filepath, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert your background image to Base64
background_image_base64 = img_to_base64('AstraZeneca_England.png')

# Custom styling for a pharmaceutical theme
st.markdown(
    f"""
    <style>
    /* Custom CSS to inject into Streamlit */

    /* Apply background image to the entire app container */
    [data-testid="stAppViewContainer"] {{
        background-image: url('data:image/png;base64,{background_image_base64}'); /* Embedded Base64 image */
        background-size: 50%; /* Adjust size as needed, e.g., 'cover', 'contain', or percentage */
        background-position: bottom right; /* Position the image */
        background-repeat: no-repeat; /* Do not repeat the image */
        background-attachment: fixed; /* Keep background fixed when scrolling */
        position: relative; /* Needed for z-index of pseudo-element */
    }}

    /* Apply a subtle white overlay *behind* the main content area, but *over* the background image */
    /* This makes the image subtle without affecting content opacity */
    [data-testid="stAppViewContainer"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.7); /* Light white overlay to make background image more subtle */
        z-index: -1; /* Ensure it's behind the actual content */
    }}

    /* Ensure main content area is opaque white so text is readable */
    .main {{
        background-color: #FFFFFF; /* Solid white background for content to prevent blurriness */
        padding: 20px;
        border-radius: 10px;
        margin: 0 auto;
        max-width: 1200px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        position: relative; /* Ensure it stacks above the ::before pseudo-element */
        z-index: 0; /* Higher than -1 of ::before */
    }}

    .stApp {{max-width: 1200px; margin: 0 auto;}}
    .header-style {{color: #5C007C; font-family: 'Arial Black', sans-serif; font-size: 3em; margin-bottom: 0px;}} /* AstraZeneca Purple */
    .subheader-style {{color: #5C007C; font-family: 'Arial', sans-serif; font-size: 1.5em; margin-top: 5px; margin-bottom: 20px;}} /* AstraZeneca Purple */
    .sidebar .sidebar-content {{background-color: #F8F8F8;}} /* Very light grey for sidebar background */
    .stButton>button {{background-color: #5C007C; color: white; border-radius: 5px; border: none; padding: 10px 20px; font-size: 16px;}} /* AstraZeneca Purple */
    .stButton>button:hover {{background-color: #4A0065;}} /* Darker purple on hover */
    .stSelectbox, .stRadio, .stNumberInput {{margin-bottom: 15px;}}
    .kpi-metric {{font-size: 2.5em; color: #28A745; font-weight: bold;}} /* Retaining green for KPI for good contrast */
    .atc-info-title {{font-weight: bold; color: yellow; margin-top: 10px;}} /* Changed from white to AstraZeneca Purple for readability */
    .model-metrics {{font-size: 0.9em; color: #6c757d;}} /* Grey for small text */

    /* Adjust for Streamlit's internal spacing */
    .block-container {{padding-top: 2rem; padding-bottom: 2rem;}}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="AstraZeneca Demand Forecasting", layout="centered", initial_sidebar_state="expanded")

# --- Top Left Logo ---
st.image('Astrazeneca_Logo.png', width=150) # Ensure this file is in the same directory as app.py

# --- Main Application Title ---
st.markdown("<h1 class='header-style'>AstraZeneca Global Demand Forecasting Command Center</h1>", unsafe_allow_html=True)
st.write("This application provides a user-friendly interface to forecast demand for various AstraZeneca ATC (Anatomical Therapeutic Chemical) classifications. Please select your criteria to predict future demand using two different machine learning models.")

# --- Sidebar for ATC Information ---
st.sidebar.markdown('<div class="sidebar-header-style"><h2>ATC Classification Info</h2></div>', unsafe_allow_html=True)

# Dynamically identified ATC columns: ['M01AB', 'M01AE', 'R03', 'N05B', 'N05C'] from notebook
ATC_INFO = {
    "M01AB": {
        "name": "Anti-inflammatory and antirheumatic products, non-steroids, acetic acid derivatives and related substances",
        "description": "Used for conditions like arthritis, pain, and inflammation.",
        "astra_drug": "Vimovo (naproxen/esomeprazole) - for osteoarthritis, rheumatoid arthritis, and ankylosing spondylitis."
    },
    "M01AE": {
        "name": "Anti-inflammatory and antirheumatic products, non-steroids, propionic acid derivatives",
        "description": "Common NSAIDs for pain and inflammation management.",
        "astra_drug": "Similar to M01AB, often includes common NSAIDs like ibuprofen, though AstraZeneca focuses more on specific indications."
    },
    "R03": {
        "name": "Drugs for obstructive airway diseases",
        "description": "Treats conditions like asthma and COPD, improving breathing.",
        "astra_drug": "Symbicort (budesonide/formoterol) - for asthma and COPD; Pulmicort (budesonide) - for asthma."
    },
    "N02BA": {
        "name": "Other analgesics and antipyretics",
        "description": "Includes acetylsalicylic acid and derivatives, often used for pain relief and fever reduction.",
        "astra_drug": "Various pain management solutions."
    },
    "N02BE": {
        "name": "Pyrazolones",
        "description": "Includes paracetamol (acetaminophen) and combinations, widely used for pain and fever.",
        "astra_drug": "Common over-the-counter pain relievers."
    },
    "N05B": {
        "name": "Anxiolytics",
        "description": "Medications used to treat anxiety disorders.",
        "astra_drug": "Various anti-anxiety medications."
    },
    "N05C": {
        "name": "Hypnotics and sedatives",
        "description": "Medications primarily used to induce sleep or sedation.",
        "astra_drug": "Sleep-aid related products."
    },
    "R06": {
        "name": "Antihistamines for systemic use",
        "description": "Medications used to relieve symptoms of allergies.",
        "astra_drug": "Allergy relief medications."
    }
}

# Define actual_atc_columns from the ATC_INFO dictionary keys for consistency
actual_atc_columns = sorted(list(ATC_INFO.keys()))

selected_atc_info_key = st.sidebar.selectbox(
    "Select an ATC Code to learn more:",
    list(ATC_INFO.keys())
)

st.sidebar.markdown(f"<p class='atc-info-title'>{selected_atc_info_key}: {ATC_INFO[selected_atc_info_key]['name']}</p>", unsafe_allow_html=True)
st.sidebar.write(f'**AstraZeneca Example:** {ATC_INFO[selected_atc_info_key]["astra_drug"]}')


# --- Main Application Inputs and Controls ---
st.markdown("<h2 class='subheader-style'>Forecast Parameters</h2>", unsafe_allow_html=True) # New subheader for clarity

# User Inputs
atc_classification = st.selectbox(
    "Select ATC Classification:",
    list(ATC_INFO.keys())
)

forecast_granularity = st.radio(
    "Forecast By:",
    ("Year Only", "Year + Month")
)

# Dynamic Year and Month Selection
current_year = datetime.now().year
year_options = list(range(2023, current_year + 5)) # Forecast up to 5 years into the future

selected_year = st.selectbox(
    "Select Year:",
    year_options
)

selected_month = None
if forecast_granularity == "Year + Month":
    month_options = {datetime(2000, m, 1).strftime('%B'): m for m in range(1, 13)}
    selected_month_name = st.selectbox(
        "Select Month:",
        list(month_options.keys())
    )
    selected_month = month_options[selected_month_name]

# Model Selection
st.subheader("Model Selection")
model_choice = st.radio(
    "Choose a model for prediction:",
    (
        f"Linear Regression (Macro/Long-Term) - RMSE: {LR_RMSE:.4f}, R^2: {LR_R2:.4f}",
        f"Random Forest (Micro/Short-Term) - RMSE: {RF_RMSE:.4f}, R^2: {RF_R2:.4f}"
    ),
    index=0 # Default to Linear Regression for macro/long-term
)


selected_model = lr_model if "Linear Regression" in model_choice else rf_model

# Prediction Logic
if st.button("Generate Forecast"):
    st.markdown("### Demand Forecast Results")

    # Determine the date range for prediction based on user selection
    if forecast_granularity == "Year Only":
        start_date = datetime(selected_year, 1, 1)
        end_date = datetime(selected_year, 12, 31)
    else: # Year + Month
        start_date = datetime(selected_year, selected_month, 1)
        # Calculate end of month
        if selected_month == 12:
            end_date = datetime(selected_year, 12, 31)
        else:
            end_date = datetime(selected_year, selected_month + 1, 1) - timedelta(days=1)

    # Create a DataFrame for future dates
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame(index=future_dates)

    future_df['Year'] = future_df.index.year
    future_df['Month'] = future_df.index.month
    future_df['DayOfWeek'] = future_df.index.dayofweek

    # Simulate lag and rolling mean features for future dates.
    # To ensure consistent predictions for the same input, these placeholder values are now fixed.
    # In a real-world scenario, these would be derived from actual historical data.
    for col in actual_atc_columns:
        future_df[f'{col}_Lag1'] = 350.0  # Fixed placeholder for Lag1 (mid-point of 50-700)
        future_df[f'{col}_Lag2'] = 350.0  # Fixed placeholder for Lag2
        future_df[f'{col}_Lag7'] = 350.0  # Fixed placeholder for Lag7
        future_df[f'{col}_RollingMean7'] = 410.0 # Fixed placeholder for RollingMean7 (mid-point of 50-700)

    # Ensure feature columns match model's training features (X_cols from Colab pipeline)
    # The order must be exactly as during training: ['Year', 'Month', 'DayOfWeek', M01AB_Lag1, M01AE_Lag1, R03_Lag1, M01AB_Lag2, ...]
    model_feature_cols = ['Year', 'Month', 'DayOfWeek'] + \
         [f'{col}_Lag1' for col in actual_atc_columns] + \
         [f'{col}_Lag2' for col in actual_atc_columns] + \
         [f'{col}_Lag7' for col in actual_atc_columns] + \
         [f'{col}_RollingMean7' for col in actual_atc_columns]

    # Filter future_df to only include the features the model expects and ensure order
    # If StandardScaler was used in training, X_future might need to be scaled here
    X_future = future_df[model_feature_cols]

    # If the selected model is Linear Regression, scale X_future
    if "Linear Regression" in model_choice:
        # The scaler is not saved, so we re-fit it. This is a simplification.
        # In a real-world app, the trained scaler object would also be saved and loaded.
        # For now, we will assume no scaling for prediction in the app as the scaler was not saved.
        pass # No scaling in app for simplicity, as scaler not saved.

    # Make predictions using the selected model
    raw_predictions = selected_model.predict(X_future)

    # The model was trained to predict all smoothed target columns simultaneously.
    # We need to extract the prediction for the selected `atc_classification`. 
    # We rely on ATC_INFO.keys() having the same order as actual_atc_columns for this indexing.
    target_col_index = actual_atc_columns.index(atc_classification) # Use sorted actual_atc_columns for indexing
    predicted_demand_daily = raw_predictions[:, target_col_index]

    # Aggregate predictions
    total_predicted_demand = np.sum(predicted_demand_daily)

    # Display KPI
    st.markdown(
        f"Predicted AstraZeneca Demand for **{atc_classification}** for {'the year ' + str(selected_year) if forecast_granularity == 'Year Only' else datetime(selected_year, selected_month, 1).strftime('%B %Y')}: <span class='kpi-metric'>{int(total_predicted_demand):,} Units</span>",
        unsafe_allow_html=True
    )

    # Basic Visualization: Daily predicted demand
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Demand': predicted_demand_daily
    })
    prediction_df.set_index('Date', inplace=True)

    st.line_chart(prediction_df['Predicted Demand'])
    st.write(f"*Daily predicted demand for {atc_classification} (smoothed) using {model_choice.split('(')[0].strip()} model.*")

    st.markdown("---<br>_Disclaimer: This forecast is based on a trained model and data for demonstration purposes._")

# Moved the descriptions after the main inputs and forecast button
st.markdown("<h2 class='subheader-style'>Key Components and Model Information</h2>", unsafe_allow_html=True)
st.markdown(
    """
    ### Key Components and Functionality:

    1.  **Sidebar for ATC Information:** On the left, you'll find an educational section about ATC classifications. Select an ATC code to learn about its therapeutic category, description, and an example of a relevant AstraZeneca drug.

    2.  **User Inputs for Forecasting:**
        *   **ATC Classification:** Choose the specific ATC code you want to forecast.
        *   **Forecast Granularity:** Decide whether to forecast for a whole year or a specific month within a year.
        *   **Year/Month Selection:** Select the desired future period for your forecast.

    3.  **Model Selection & Explanation:
        *   You can choose between a **Linear Regression** model and a **Random Forest** model.
        *   Their performance metrics (RMSE and R^2) are displayed for transparency.

    4.  **Demand Forecast Results:** After clicking "Generate Forecast," the application will display the predicted demand for your selected criteria, including a visual trend over the forecast period.

    ### Understanding the Models and Potential Pitfalls:

    *   **Linear Regression (Macro/Long-Term):**
        *   **How it works:** This model establishes a linear relationship between input features (like Year, Month, Lag features) and the smoothed demand. It's excellent for identifying and extending overall trends.
        *   **Strengths:** Simple, interpretable, and generally robust for long-term forecasting as it effectively extrapolates based on historical trends. Less prone to overfitting noise.
        *   **Why it might predict higher units:** If the training data shows an upward trend, Linear Regression will continue that trend linearly into the future, even with simplified placeholder features. It excels at extending linear growth or decline.
        *   **Pitfalls:** Assumes a linear relationship, which might not always hold true. Sensitive to outliers. Its predictions are an extrapolation based on observed linearity, which may not always reflect real-world non-linear shifts.

    *   **Random Forest (Micro/Short-Term):**
        *   **How it works:** An ensemble of decision trees, this model captures complex, non-linear relationships and interactions. It makes predictions by averaging the outputs of many individual decision trees.
        *   **Strengths:** Highly flexible, can capture intricate patterns, and generally performs well on short-term, detailed predictions.
        *   **Why it might predict lower units:** Random Forests are generally **poor at extrapolation**. They predict by finding similar patterns in the training data and averaging target values from those patterns. When given simplified, fixed placeholder features for future dates (as in this app), the Random Forest tends to predict values within the range it has already seen during training, rather than extending trends. It interpolates within its learned data space.
        *   **Pitfalls:** Can be prone to overfitting if not properly tuned. Less interpretable than linear models. The main pitfall in this application context is its inability to extrapolate; if you forecast far into the future with placeholder features, its predictions might appear conservative or "lower" because it's effectively predicting an average of similar historical scenarios rather than extending a trend.

    *   **Feature Simplification in App:** For demonstration purposes, future lag and rolling mean features are populated with fixed placeholder values. In a real-world production system, these would need to be dynamically calculated from actual historical data up to the prediction start date. This simplification is a key reason for the different behaviors observed between the models, particularly the Random Forest's more conservative predictions.

    *   **Scaler Serialization:** The `StandardScaler` used for Linear Regression during training is *not* saved or loaded in this app. For a robust production solution, the fitted `StandardScaler` would need to be saved alongside the model and used to transform input features in the app before prediction. This is a current limitation.
    """
)
st.markdown(
    """
    **Why two models?**
    *   **Linear Regression (Macro/Long-Term):** Often performs well for longer-term trends and overall patterns. Its simplicity can make it robust for forecasting general direction over extended periods, even with some data volatility. It's less prone to overfitting noise present in short-term fluctuations.
    *   **Random Forest (Micro/Short-Term):** Excels at capturing complex, non-linear relationships and interactions between features, making it highly effective for short-term, detailed predictions. However, its complexity can sometimes lead to overfitting if not carefully tuned, and it might not generalize as well to very long-term, unseen patterns compared to simpler models like Linear Regression. When forecasting with simplified placeholder features for future dates, it may also produce predictions that appear lower or less extrapolated than linear models.
    """
)