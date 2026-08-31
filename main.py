# imports
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# defining items
items_list = ["N95_Masks","IV_Fluids","Antibiotics","Inhalers","Insulin_Vials","Painkillers"]

# import rag from rag.py
try:
    from rag import rag_chain
except Exception as e:
    rag_chain = None

# page configuration
st.set_page_config(
    page_title="StockPulse | Healthcare Supply & Procurement",
    layout="wide",
    initial_sidebar_state="expanded"
)

#------STYLING------

st.markdown("""
<style>

    /* =========================
       GLOBAL APPLICATION
       ========================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #f7f5fc 0%,
            #f1eef9 55%,
            #ebe7f5 100%
        ) !important;
        color: #30303b;
    }

    .main {
        background-color: transparent;
    }

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background-color: #faf9fd !important;
        border-right: 1px solid #ddd7eb;
    }

    [data-testid="stSidebar"] * {
        color: #3f3a4d;
    }

    /* =========================
       TYPOGRAPHY
       ========================= */

    h1, h2, h3 {
        color: #30264a !important;
        font-family: "Arial", sans-serif;
        font-weight: 600;
        letter-spacing: -0.2px;
    }

    h1 {
        font-size: 2rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
    }

    p, label, span {
        font-family: "Arial", sans-serif;
    }

    /* =========================
       METRIC CARDS
       ========================= */

    .metric-card {
        background-color: #ffffff;
        padding: 18px 20px;
        border-radius: 7px;
        border: 1px solid #e1dceb;
        border-left: 4px solid #7355a8;
        box-shadow: 0 1px 3px rgba(48, 38, 74, 0.07);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-sizing: border-box;
    }

    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }

    [data-testid="column"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    [data-testid="column"] > div > div {
        flex: 1;
        display: flex;
        flex-direction: column;
    }

    /* =========================
       ALERTS / STATUS
       ========================= */

    .alert-red {
        background-color: #fff6f6;
        border-left: 4px solid #d64545;
        padding: 11px 13px;
        border-radius: 5px;
        color: #8f2525;
        margin-bottom: 10px;
        font-weight: 500;
    }

    .alert-orange {
        background-color: #fff8f2;
        border-left: 4px solid #d9772b;
        padding: 11px 13px;
        border-radius: 5px;
        color: #914b16;
        margin-bottom: 10px;
        font-weight: 500;
    }

    .alert-yellow {
        background-color: #fffdf2;
        border-left: 4px solid #c49a19;
        padding: 11px 13px;
        border-radius: 5px;
        color: #765d0b;
        margin-bottom: 10px;
        font-weight: 500;
    }

    /* =========================
       RAG / COMPLIANCE PANEL
       ========================= */

    .rag-box {
        background-color: #ffffff;
        border: 1px solid #ddd7eb;
        border-left: 4px solid #7355a8;
        padding: 16px 18px;
        border-radius: 7px;
        box-shadow: 0 1px 3px rgba(48, 38, 74, 0.06);
        margin-top: 15px;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border-radius: 5px;
        border: 1px solid #7355a8;
        background-color: #7355a8;
        color: white;
        font-weight: 500;
        padding: 0.45rem 1rem;
        box-shadow: none;
    }

    .stButton > button:hover {
        background-color: #604590;
        border-color: #604590;
        color: white;
    }

    /* =========================
       INPUTS
       ========================= */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stDateInput input {
        border-radius: 5px;
        border: 1px solid #d5d0df;
        background-color: #ffffff;
    }

    /* =========================
       DATA TABLES
       ========================= */

    [data-testid="stDataFrame"] {
        border: 1px solid #ded9e8;
        border-radius: 6px;
        background-color: #ffffff;
    }

    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #ddd8e6 !important;
    }

</style>
""", unsafe_allow_html=True)


# data loading and model training
@st.cache_data
def load_data():
    df = pd.read_csv("StockPulse_synthetic_data.csv")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="mixed",
        dayfirst=True
    )

    df = df.sort_values(
        ["Facility_ID", "Timestamp"]
    ).reset_index(drop=True)

    return df


df_global = load_data()


# weather normalization
WEATHER_TYPES = [
    "None",
    "Heatwave",
    "Sandstorm/Dust_Storm"
]


def normalize_weather(value):
    if pd.isna(value):
        return "None"

    value = str(value).strip()

    if value in ["", "Normal", "None"]:
        return "None"

    if value in ["Sandstorm", "Dust_Storm", "Sandstorm/Dust_Storm"]:
        return "Sandstorm/Dust_Storm"

    if value == "Heatwave":
        return "Heatwave"

    return value


# Normalize weather consistently for training and prediction
df_global["Weather_Type"] = df_global["Weather_Type"].apply(
    normalize_weather
)


# model training and evaluation
@st.cache_resource(show_spinner="Training Multivariate Forecasting Models...")
def train_models_in_memory(df):

    df_proc = df.copy()

    # Normalize weather categories consistently
    df_proc["Weather_Type"] = df_proc["Weather_Type"].apply(
        normalize_weather
    )

    # Sort before creating historical features
    df_proc = df_proc.sort_values(
        ["Facility_ID", "Timestamp"]
    ).reset_index(drop=True)

    # ------------------------------------------------
    # Historical features
    # IMPORTANT:
    # No bfill() is used here so future observations cannot leak backwards into earlier rows.
    # ------------------------------------------------
    for item in items_list:

        df_proc[f"{item}_Lag_7d"] = (
            df_proc.groupby("Facility_ID")[f"Units_{item}"]
            .shift(7)
        )

        df_proc[f"{item}_Rolling_7d"] = (
            df_proc.groupby("Facility_ID")[f"Units_{item}"]
            .transform(
                lambda x:
                    x.rolling(
                        window=7,
                        min_periods=7
                    )
                    .mean()
                    .shift(1)
            )
        )

    # Encode categorical variables
    df_encoded = pd.get_dummies(
        df_proc,
        columns=["Facility_ID", "Weather_Type"],
        dtype=int
    )

    base_features = [
        "Local_Temp_C",
        "Seasonal_Flu_Rate",
        "Weather_Alert_Flag",
        "Admitted_Patients"
    ]

    facility_cols = [
        c for c in df_encoded.columns
        if c.startswith("Facility_ID_")
    ]

    weather_cols = [
        c for c in df_encoded.columns
        if c.startswith("Weather_Type_")
    ]

    models = {}
    evaluation_metrics = {}

    # Train one model per medical item
    for item in items_list:

        target_col = f"Target_{item}_7d_Ahead"

        required_columns = (
            ["Timestamp"]
            + base_features
            + facility_cols
            + weather_cols
            + [
                f"{item}_Lag_7d",
                f"{item}_Rolling_7d"
            ]
            + [target_col]
        )

        # Keep only rows where all required information genuinely exists.
        item_df = df_encoded[
            required_columns
        ].dropna(
            subset=[
                f"{item}_Lag_7d",
                f"{item}_Rolling_7d",
                target_col
            ]
        ).copy()

        X_item = item_df[
            base_features
            + facility_cols
            + weather_cols
            + [
                f"{item}_Lag_7d",
                f"{item}_Rolling_7d"
            ]
        ]

        y = item_df[target_col]

        # ------Time-based validation split------

        # Sort chronologically before splitting
        item_df = item_df.sort_values("Timestamp").copy()

        # Use a chronological 75/25 split.

        unique_dates = np.sort(
            item_df["Timestamp"].unique()
        )

        if len(unique_dates) >= 20:

            split_position = int(
                len(unique_dates) * 0.75
            )

            split_date = unique_dates[split_position]

            # Training = earlier dates
            train_mask = (
                item_df["Timestamp"] < split_date
            )

            # Testing = later dates
            test_mask = (
                item_df["Timestamp"] >= split_date
            )

            train_data = item_df.loc[train_mask]
            test_data = item_df.loc[test_mask]

            X_item = item_df[
                base_features
                + facility_cols
                + weather_cols
                + [
                    f"{item}_Lag_7d",
                    f"{item}_Rolling_7d"
                ]
            ]

            y = item_df[target_col]

            X_train = X_item.loc[train_data.index]
            X_test = X_item.loc[test_data.index]

            y_train = y.loc[train_data.index]
            y_test = y.loc[test_data.index]

            # ------Train validation model------
            validation_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            validation_model.fit(
                X_train,
                y_train
            )

            # Generate predictions on unseen future period
            test_predictions = validation_model.predict(
                X_test
            )

            # ------Evaluation metrics------
            mae = mean_absolute_error(
                y_test,
                test_predictions
            )
            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    test_predictions
                )
            )
            r2 = r2_score(
                y_test,
                test_predictions
            )

            # MAPE
            actual_values = np.asarray(y_test)
            predicted_values = np.asarray(test_predictions)

            non_zero_mask = actual_values != 0

            if np.any(non_zero_mask):

                mape = np.mean(
                    np.abs(
                        (
                            actual_values[non_zero_mask]
                            - predicted_values[non_zero_mask]
                        )
                        / actual_values[non_zero_mask]
                    )
                ) * 100

            else:
                mape = np.nan

            # ------Store metrics------

            evaluation_metrics[item] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "MAPE": (
                    float(mape)
                    if not np.isnan(mape)
                    else None
                ),
                "Train_Start": str(
                    train_data["Timestamp"].min().date()
                ),
                "Train_End": str(
                    train_data["Timestamp"].max().date()
                ),
                "Test_Start": str(
                    test_data["Timestamp"].min().date()
                ),
                "Test_End": str(
                    test_data["Timestamp"].max().date()
                )
            }

        else:

            evaluation_metrics[item] = {
                "MAE": None,
                "RMSE": None,
                "R2": None,
                "MAPE": None,
                "Train_Start": None,
                "Train_End": None,
                "Test_Start": None,
                "Test_End": None
            }

        # Final production model after evaluating the model, retrain using all valid historical observations.
        final_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        final_model.fit(
            X_item,
            y
        )

        models[item] = final_model

    return models, evaluation_metrics


models_dict, model_metrics = train_models_in_memory(
    df_global
)


# ------Date-aware feature preparation------
def get_historical_features(
    df_facility,
    target_date,
    item_name
):
    """
    Creates lag and rolling features based on the actual
    calendar date rather than dataframe row position.

    No future observations are used.
    """

    df_facility = df_facility.copy()

    df_facility["Timestamp"] = pd.to_datetime(
        df_facility["Timestamp"]
    )

    df_facility = df_facility.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    target_date = pd.Timestamp(target_date).normalize()

    # Exact 7-calendar-day lag
    lag_date = target_date - pd.Timedelta(days=7)

    lag_rows = df_facility[
        df_facility["Timestamp"].dt.normalize()
        == lag_date
    ]

    if not lag_rows.empty:

        lag_7d = float(
            lag_rows.iloc[-1][
                f"Units_{item_name}"
            ]
        )

    else:

        # use the most recent observation on or before the required lag date.
        prior_lag_rows = df_facility[
            df_facility["Timestamp"].dt.normalize()
            <= lag_date
        ]

        if not prior_lag_rows.empty:

            lag_7d = float(
                prior_lag_rows.iloc[-1][
                    f"Units_{item_name}"
                ]
            )

        else:

            # If there is no earlier historical observation, use the available historical average.
            historical_values = df_facility[
                f"Units_{item_name}"
            ].dropna()

            lag_7d = float(
                historical_values.mean()
            ) if not historical_values.empty else 0.0

    # Previous 7 calendar days rolling average
    rolling_start = target_date - pd.Timedelta(days=7)

    rolling_rows = df_facility[
        (df_facility["Timestamp"].dt.normalize() >= rolling_start)
        &
        (df_facility["Timestamp"].dt.normalize() < target_date)
    ]

    if not rolling_rows.empty:

        rolling_7d = float(
            rolling_rows[
                f"Units_{item_name}"
            ].mean()
        )

    else:

        # Historical-only fallback
        prior_rows = df_facility[
            df_facility["Timestamp"].dt.normalize()
            < target_date
        ]

        if not prior_rows.empty:

            rolling_7d = float(
                prior_rows[
                    f"Units_{item_name}"
                ].tail(7).mean()
            )

        else:

            rolling_7d = lag_7d

    return lag_7d, rolling_7d


def prepare_model_features(
    df_facility,
    target_date,
    item_name,
    model_obj,
    simulated_row
):
    """
    Builds prediction features for the selected calendar date.
    Historical lag features are obtained from actual dates,
    not row positions.
    """

    current_row = simulated_row

    base_feats = {
        "Local_Temp_C": current_row["Local_Temp_C"],
        "Seasonal_Flu_Rate": current_row["Seasonal_Flu_Rate"],
        "Weather_Alert_Flag": current_row["Weather_Alert_Flag"],
        "Admitted_Patients": current_row["Admitted_Patients"]
    }

    # Facility categories taken from the model's training data to ensure consistency
    facility_categories = [
        col.replace("Facility_ID_", "")
        for col in model_obj.feature_names_in_
        if col.startswith("Facility_ID_")
    ]

    for facility in facility_categories:

        base_feats[
            f"Facility_ID_{facility}"
        ] = int(
            current_row["Facility_ID"] == facility
        )

    # Weather categories match training exactly
    weather_value = normalize_weather(
        current_row["Weather_Type"]
    )

    weather_categories = [
        col.replace("Weather_Type_", "")
        for col in model_obj.feature_names_in_
        if col.startswith("Weather_Type_")
    ]

    for weather in weather_categories:

        base_feats[
            f"Weather_Type_{weather}"
        ] = int(
            weather_value == weather
        )

    # Date-aware lag and rolling features
    lag_7d, rolling_7d = get_historical_features(
        df_facility,
        target_date,
        item_name
    )

    base_feats[
        f"{item_name}_Lag_7d"
    ] = lag_7d

    base_feats[
        f"{item_name}_Rolling_7d"
    ] = rolling_7d

    X_df = pd.DataFrame(
        [base_feats]
    )

    # Guarantee exactly the feature structure expected by the trained Random Forest
    for col in model_obj.feature_names_in_:

        if col not in X_df.columns:
            X_df[col] = 0

    X_df = X_df[
        model_obj.feature_names_in_
    ]

    return X_df


# 5------. SESSION STATE------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "hospital" not in st.session_state:
    st.session_state.hospital = "HOSP_RUH_01"

if "username" not in st.session_state:
    st.session_state.username = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "predicted_orders" not in st.session_state:
    st.session_state.predicted_orders = {}

if "safety_stock" not in st.session_state:
    st.session_state.safety_stock = {}

if "net_reorder_qty" not in st.session_state:
    st.session_state.net_reorder_qty = {}

if "item_feature_drivers" not in st.session_state:
    st.session_state.item_feature_drivers = {}

if "rejection_logs" not in st.session_state:
    st.session_state.rejection_logs = []

if "order_status" not in st.session_state:
    st.session_state.order_status = "Pending"

if "user_inventory" not in st.session_state:
    st.session_state.user_inventory = {
    "N95_Masks": 100,
    "IV_Fluids": 250,
    "Antibiotics": 400,
    "Inhalers": 80,
    "Insulin_Vials": 90,
    "Painkillers": 100
}

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = model_metrics


# ==========================================
# VIEW 1: LOGIN PAGE
# ==========================================

if not st.session_state.logged_in:

    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown(
            "<h1 style='text-align: center;'>💊 StockPulse AI</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align: center; color: #6b7280;'>"
            "Standardized AI-Driven Inventory Forecasting & Procurement"
            "</p>",
            unsafe_allow_html=True
        )

        with st.form("login_form"):

            st.markdown(
                "### Secure Staff Portal Login"
            )

            hospital_choice = st.selectbox(
                "Choose Healthcare Facility",
                [
                    "HOSP_RUH_01 (Riyadh Central)",
                    "HOSP_JED_02 (Jeddah General)",
                    "HOSP_DMM_03 (Dammam Medical City)"
                ]
            )

            username = st.text_input(
                "Username",
                placeholder="e.g., Dr. Sarah"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••"
            )

            submit_btn = st.form_submit_button(
                "Access Portal 🚀",
                use_container_width=True
            )

            if submit_btn:

                st.session_state.logged_in = True

                st.session_state.hospital = (
                    hospital_choice.split(" ")[0]
                )

                st.session_state.username = (
                    username
                    if username
                    else "Chief Pharmacist"
                )

                st.rerun()

    st.stop()


# ==========================================
# SIDEBAR SETUP
# ==========================================

st.sidebar.markdown(
    "# StockPulse AI"
)

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Orders & Procurement"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.messages = []

    st.rerun()

st.sidebar.markdown("---")


# ------Dashboard inventory controls------
if nav_selection == "Dashboard":

    st.sidebar.markdown(
        "### ⚙️ Inventory Adjustments"
    )

    for item in items_list:

        st.session_state.user_inventory[item] = (
            st.sidebar.number_input(
                f"On-Hand: {item}",
                min_value=0,
                max_value=5000,
                value=st.session_state.user_inventory.get(
                    item,
                    100
                ),
                step=10
            )
        )

    st.sidebar.markdown("---")


st.sidebar.markdown(
    "### 🏥 Facility Info"
)

st.sidebar.text(
    f"Active: {st.session_state.get('hospital', 'HOSP_RUH_01')}"
)


# ------INVENTORY / FORECAST CONFIGURATION------

# Kept as the existing 20% rule
FORECAST_BUFFER_PCT = 0.20


# ------MAIN APPLICATION DATA------

df_hosp = df_global[
    df_global["Facility_ID"]
    == st.session_state.hospital
].sort_values(
    "Timestamp"
).reset_index(drop=True)


# ------SIMULATED OPERATIONAL ROW------

def generate_simulated_row(
    target_dt,
    base_r
):
    facility_id = base_r["Facility_ID"]

    day_of_year = target_dt.timetuple().tm_yday
    month = target_dt.month
    day_of_week = target_dt.weekday()

    date_int = (
        target_dt.year * 10000
        + target_dt.month * 100
        + target_dt.day
    )

    fac_code = sum(
        ord(c)
        for c in facility_id
    )

    rng = np.random.RandomState(
        seed=date_int + fac_code
    )

    base_temp = (
        31.0
        + 11.0
        * np.sin(
            2 * np.pi
            * (day_of_year - 105)
            / 365
        )
    )

    temp_noise = rng.normal(
        0,
        1.8
    )

    sim_temp = float(
        np.clip(
            base_temp + temp_noise,
            12.0,
            50.0
        )
    )

    is_summer = month in [
        6,
        7,
        8,
        9
    ]

    is_spring = month in [
        3,
        4,
        5
    ]

    sim_weather = "None"
    temp_boost = 0.0

    if (
        is_summer
        and sim_temp > 38.0
        and rng.rand() < 0.15
    ):

        sim_weather = "Heatwave"

        temp_boost = rng.uniform(
            4.0,
            7.0
        )

    elif (
        is_spring
        and rng.rand() < 0.12
    ):

        sim_weather = "Sandstorm/Dust_Storm"

    sim_temp = round(
        float(
            np.clip(
                sim_temp + temp_boost,
                12.0,
                52.0
            )
        ),
        1
    )

    sim_alert = int(
        sim_weather != "None"
    )

    flu_base = (
        65.0
        + 30.0
        * np.cos(
            2 * np.pi
            * (day_of_year - 15)
            / 365
        )
    )

    flu_noise = rng.normal(
        0,
        4.0
    )

    sim_flu = round(
        float(
            np.clip(
                flu_base + flu_noise,
                5.0,
                100.0
            )
        ),
        2
    )

    facility_baselines = {
        "HOSP_RUH_01": 220,
        "HOSP_JED_02": 150,
        "HOSP_DMM_03": 95
    }

    base_capacity = facility_baselines.get(
        facility_id,
        150
    )

    flu_surge = (
        sim_flu / 100.0
    ) * (
        base_capacity * 0.25
    )

    weather_surge = (
        base_capacity * 0.20
        if sim_weather == "Heatwave"
        else (
            base_capacity * 0.30
            if sim_weather == "Sandstorm/Dust_Storm"
            else 0
        )
    )

    dow_multipliers = [
        1.05,
        1.08,
        1.06,
        1.02,
        0.95,
        0.85,
        0.88
    ]

    dow_mult = dow_multipliers[
        day_of_week
    ]

    patient_noise = rng.normal(
        0,
        base_capacity * 0.05
    )

    total_patients = (
        base_capacity
        + flu_surge
        + weather_surge
        + patient_noise
    ) * dow_mult

    sim_patients = int(
        max(
            20,
            round(total_patients)
        )
    )

    sim_row = base_r.copy()

    sim_row["Timestamp"] = pd.Timestamp(
        target_dt
    )

    sim_row["Local_Temp_C"] = sim_temp
    sim_row["Seasonal_Flu_Rate"] = sim_flu
    sim_row["Admitted_Patients"] = sim_patients
    sim_row["Weather_Type"] = sim_weather
    sim_row["Weather_Alert_Flag"] = sim_alert

    item_factors = {
        "N95_Masks": 1.8,
        "IV_Fluids": 2.4,
        "Antibiotics": 1.2,
        "Inhalers": 0.8,
        "Insulin_Vials": 0.6,
        "Painkillers": 2.1
    }

    for item, factor in item_factors.items():

        item_boost = 1.0

        if (
            item == "Inhalers"
            and sim_weather == "Sandstorm/Dust_Storm"
        ):
            item_boost *= 2.2

        if (
            item == "IV_Fluids"
            and sim_weather == "Heatwave"
        ):
            item_boost *= 1.8

        if (
            item in [
                "N95_Masks",
                "Antibiotics"
            ]
            and sim_flu > 70
        ):
            item_boost *= 1.5

        item_units = (
            sim_patients
            * factor
            * item_boost
            * rng.uniform(
                0.9,
                1.1
            )
        )

        sim_row[
            f"Units_{item}"
        ] = int(
            max(
                5,
                round(item_units)
            )
        )

    return sim_row


# ==========================================
# SECTION A: DASHBOARD VIEW
# ==========================================

if nav_selection == "Dashboard":

    st.markdown(
        "#  StockPulse Executive Dashboard"
    )

    st.markdown(
    f"AI-powered predictive analytics and autonomous "
    f"stock monitoring for *{st.session_state.hospital}*."
)

    min_date = datetime(
        2026,
        1,
        1
    ).date()

    max_date = datetime(
        2026,
        12,
        31
    ).date()

    today_real = datetime.now().date()

    default_date = max(
        min_date,
        min(
            today_real,
            max_date
        )
    )

    selected_date = st.date_input(
        "📅 Select Operational Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help=(
            "Select any target operational date in 2026 "
            "to generate predictive AI forecasts."
        )
    )

    latest_idx = len(df_hosp) - 1

    base_row = df_hosp.iloc[
        latest_idx
    ]

    simulated_target_row = generate_simulated_row(
        selected_date,
        base_row
    )

    row_data = simulated_target_row

    live_predictions = {}
    predicted_orders = {}
    net_reorder_qty = {}
    current_inventory = {}
    safety_stock = {}
    item_feature_drivers = {}

    for item in items_list:

        X_input = prepare_model_features(
            df_hosp,
            selected_date,
            item,
            models_dict[item],
            simulated_target_row
        )

        pred_val = float(
            models_dict[item].predict(
                X_input
            )[0]
        )

        pred_demand = int(
            np.ceil(
                max(
                    0,
                    pred_val
                )
            )
        )

        live_predictions[item] = pred_val

        predicted_orders[item] = pred_demand

        curr_stock = st.session_state.user_inventory.get(
            item,
            100
        )

        current_inventory[item] = curr_stock

        # Existing 20% rule retained,
        # but presented as a forecast buffer.
        buffer_qty = max(
            10,
            int(
                np.ceil(
                    pred_demand
                    * FORECAST_BUFFER_PCT
                )
            )
        )

        safety_stock[item] = buffer_qty

        net_order = max(
            0,
            (
                pred_demand
                + buffer_qty
            )
            - curr_stock
        )

        net_reorder_qty[item] = net_order

        # --------------------------------------------
        # Model-level feature importance
        # --------------------------------------------
        model = models_dict[item]

        if (
            hasattr(model, "feature_importances_")
            and hasattr(model, "feature_names_in_")
        ):

            importances = (
                model.feature_importances_
            )

            feature_names = (
                model.feature_names_in_
            )

            sorted_indices = np.argsort(
                importances
            )[::-1]

            friendly_names = {
                "Admitted_Patients":
                    "Admitted Patients",

                "Seasonal_Flu_Rate":
                    "Seasonal Flu Index",

                "Local_Temp_C":
                    "Local Temperature",

                "Weather_Alert_Flag":
                    "Extreme Weather Alert",

                f"{item}_Lag_7d":
                    "7-Day Historical Demand",

                f"{item}_Rolling_7d":
                    "7-Day Rolling Average"
            }

            clean_drivers = []

            for feature_idx in sorted_indices:

                feat = feature_names[
                    feature_idx
                ]

                if (
                    "Facility" in feat
                    or "Weather_Type" in feat
                ):
                    continue

                readable_name = friendly_names.get(
                    feat,
                    feat.replace(
                        "_",
                        " "
                    )
                )

                clean_drivers.append(
                    readable_name
                )

                if len(clean_drivers) == 3:
                    break

            item_feature_drivers[item] = (
                ", ".join(clean_drivers)
            )

        else:

            item_feature_drivers[item] = (
                "Admitted Patients, "
                "Seasonal Flu Index, "
                "Local Temperature"
            )

    # ------Session state------

    st.session_state.predicted_orders = (
        predicted_orders
    )

    st.session_state.current_inventory = (
        current_inventory
    )

    st.session_state.safety_stock = (
        safety_stock
    )

    st.session_state.net_reorder_qty = (
        net_reorder_qty
    )

    st.session_state.item_feature_drivers = (
        item_feature_drivers
    )

    forecast_rows = []

    for item in items_list:

        forecast_rows.append({
            "Item_Name": item,
            "Predicted_7Day_Demand": predicted_orders[item],
            "Buffer": safety_stock[item],
            "On_Hand": current_inventory[item],
            "Net_Procurement": net_reorder_qty[item],
            "Drivers": item_feature_drivers[item]
        })

    st.session_state["forecast_df"] = (
        pd.DataFrame(forecast_rows)
    )


    # ==========================================
    # METRICS ROW
    # ==========================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.markdown(
            f"""
            <div class='metric-card'>
                <div>
                    <h4 style='margin-bottom: 5px;'>
                        Patients on {selected_date.strftime('%b %d, %Y')}
                    </h4>
                    <h2 style='margin-top: 0;'>
                        {int(row_data['Admitted_Patients'])}
                    </h2>
                </div>
                <p style='color: #10b981; font-size: 12px; margin: 10px 0 0 0;'>
                    ● Projected Hospital Load
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:

        st.markdown(
            f"""
            <div class='metric-card'>
                <div>
                    <h4 style='margin-bottom: 5px;'>
                        Local Temperature
                    </h4>
                    <h2 style='margin-top: 0;'>
                        {row_data['Local_Temp_C']} °C
                    </h2>
                </div>
                <p style='color: #f59e0b; font-size: 12px; margin: 10px 0 0 0;'>
                    ● Forecast Sensor
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:

        st.markdown(
            f"""
            <div class='metric-card'>
                <div>
                    <h4 style='margin-bottom: 5px;'>
                        Seasonal Flu Index
                    </h4>
                    <h2 style='margin-top: 0;'>
                        {row_data['Seasonal_Flu_Rate']:.1f} / 100
                    </h2>
                </div>
                <p style='color: #6366f1; font-size: 12px; margin: 10px 0 0 0;'>
                    ● Epidemic Forecast
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:

        weather_label = (
            row_data["Weather_Type"]
            if pd.notna(
                row_data["Weather_Type"]
            )
            else "Normal Conditions"
        )

        st.markdown(
            f"""
            <div class='metric-card'>
                <div>
                    <h4 style='margin-bottom: 5px;'>
                        Weather Status
                    </h4>
                    <h3 style='font-size: 18px; margin-top: 0;'>
                        {weather_label}
                    </h3>
                </div>
                <p style='color: #ef4444; font-size: 12px; margin: 10px 0 0 0;'>
                    ● Alert Flag: {int(row_data['Weather_Alert_Flag'])}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ------MODEL EVALUATION------

    with st.expander(
        "Forecast Model Performance",
        expanded=False
    ):

        metrics_rows = []

        for item in items_list:

            metrics = model_metrics.get(
                item,
                {}
            )

            metrics_rows.append({
                "Medical Item": item,
                "MAE (units)": (
                    round(metrics["MAE"], 2)
                    if metrics.get("MAE") is not None
                    else "N/A"
                ),
                "RMSE (units)": (
                    round(metrics["RMSE"], 2)
                    if metrics.get("RMSE") is not None
                    else "N/A"
                ),
                "R²": (
                    round(metrics["R2"], 3)
                    if metrics.get("R2") is not None
                    else "N/A"
                ),
                "MAPE (%)": (
                    round(metrics["MAPE"], 2)
                    if metrics.get("MAPE") is not None
                    else "N/A"
                )
            })

        st.dataframe(
            pd.DataFrame(metrics_rows),
            use_container_width=True,
            hide_index=True
        )


    # ==========================================
    # CHARTS & ALERTS
    # ==========================================

    chart_col, alert_col = st.columns([2, 1])

    with chart_col:

        st.markdown(
            "### 7-Day Demand Forecast"
        )

        selected_product = st.selectbox(
            "Select Medical Item to Inspect",
            items_list
        )

        simulated_demand = row_data[
            f"Units_{selected_product}"
        ]

        predicted_val = live_predictions[
            selected_product
        ]

        # Explicitly distinguish simulated demand
        # from the AI's +7-day prediction.
        forecast_date = selected_date + pd.Timedelta(days=7)

        x_vals = [
            f"Selected Date\n{selected_date.strftime('%b %d')}",
            f"Forecast (+7d)\n{forecast_date.strftime('%b %d')}"
        ]

        y_vals = [
            simulated_demand,
            predicted_val
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                marker_color=[
                    "#080949",
                    "#9b1d52"
                ],
                text=[
                    f"{val:.0f} units"
                    for val in y_vals
                ],
                textposition="auto",
                name="Demand"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(
                    color="#311059",
                    width=3
                ),
                marker=dict(
                    size=8
                ),
                name="Demand"
            )
        )

        fig.update_layout(
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                title="Medical Supply Units"
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            f"Demand shown for {selected_date.strftime('%b %d, %Y')} "
            f"is simulated from facility, weather, flu and patient-load "
            f"assumptions. The second value is the AI forecast for "
            f"{forecast_date.strftime('%b %d, %Y')} (+7 days)."
        )


    with alert_col:

        st.markdown(
            "### 🚨 Urgent AI Notifications"
        )

        active_warnings = 0

        for item in items_list:

            curr = st.session_state.user_inventory.get(
                item,
                100
            )

            pred = st.session_state.get(
                "predicted_orders",
                {}
            ).get(
                item,
                150
            )

            buf = st.session_state.get(
                "safety_stock",
                {}
            ).get(
                item,
                max(
                    10,
                    int(
                        pred
                        * FORECAST_BUFFER_PCT
                    )
                )
            )

            net_needed = max(
                0,
                (
                    pred
                    + buf
                )
                - curr
            )

            if net_needed > 0:

                active_warnings += 1

                st.markdown(
                    f"""
                    <div class='alert-orange'
                         style='padding: 8px 12px; margin-bottom: 6px; font-size: 14px;'>
                        <b>{item}:</b>
                        Deficit of <b>{net_needed} units</b>
                        projected for Day 7.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if active_warnings == 0:

            st.markdown(
                """
                <div class='alert-yellow'
                     style='background-color: #ecfdf5;
                            border-left-color: #10b981;
                            color: #065f46;
                            padding: 8px 12px;'>
                    <b>System Secure:</b>
                    All inventories cover the 7-day forecast.
                </div>
                """,
                unsafe_allow_html=True
            )

        if (
            pd.notna(
                row_data["Weather_Type"]
            )
            and row_data["Weather_Type"] != "None"
        ):

            st.caption(
                f"⚠️ Active Weather: "
                f"{row_data['Weather_Type']} "
                f"(Flu Index: "
                f"{row_data['Seasonal_Flu_Rate']:.1f})"
            )


# ==========================================
# SECTION B: ORDERS & PROCUREMENT VIEW
# ==========================================

elif nav_selection == "Orders & Procurement":

    st.markdown(
        "# 🛒 AI-Assisted Procurement & Supplier Orders"
    )

    st.markdown(
        "Review AI-generated purchase orders across all "
        "inventory items, verify compliance guidelines "
        "with RAG, and approve or reject draft orders."
    )

    forecast_rows = []

    for item in items_list:

        pred_demand = st.session_state.get(
            "predicted_orders",
            {}
        ).get(
            item,
            150
        )

        buffer_qty = st.session_state.get(
            "safety_stock",
            {}
        ).get(
            item,
            max(
                10,
                int(
                    np.ceil(
                        pred_demand
                        * FORECAST_BUFFER_PCT
                    )
                )
            )
        )

        curr_stock = st.session_state.get(
            "user_inventory",
            {}
        ).get(
            item,
            100
        )

        net_order = max(
            0,
            (
                pred_demand
                + buffer_qty
            )
            - curr_stock
        )

        drivers = st.session_state.get(
            "item_feature_drivers",
            {}
        ).get(
            item,
            "Admitted Patients, "
            "Seasonal Flu Index, "
            "Local Temperature"
        )

        forecast_rows.append({
            "Item_Name": item,
            "Predicted_7Day_Demand": pred_demand,
            "Buffer": buffer_qty,
            "On_Hand": curr_stock,
            "Net_Procurement": net_order,
            "Drivers": drivers
        })

    df_f = pd.DataFrame(
        forecast_rows
    )

    st.session_state["forecast_df"] = df_f


    # ------Inventory snapshot on procurement page------

    st.markdown(
        "### 📦 Inventory Snapshot"
    )

    inventory_display = df_f[
        [
            "Item_Name",
            "On_Hand",
            "Predicted_7Day_Demand",
            "Buffer",
            "Net_Procurement"
        ]
    ].rename(
        columns={
            "Item_Name": "Medical Item",
            "On_Hand": "On Hand",
            "Predicted_7Day_Demand":
                "Predicted Day +7 Demand",
            "Buffer": "Forecast Buffer",
            "Net_Procurement":
                "Net Procurement"
        }
    )

    st.dataframe(
        inventory_display,
        use_container_width=True,
        hide_index=True
    )


    col_o1, col_o2 = st.columns(2)


    # ==========================================
    # PROCUREMENT ORDER
    # ==========================================

    with col_o1:

        st.markdown(
            "### 📋 Pending Multi-Item Supplier Draft Order"
        )

        quantities_summary = {}

        manifest_display_lines = []

        for _, row in df_f.iterrows():

            iname = row.get(
                "Item_Name",
                "Unknown"
            )

            net_q = int(
                row.get(
                    "Net_Procurement",
                    0
                )
            )

            quantities_summary[iname] = net_q

            raw_drivers = row.get(
                "Drivers",
                "Seasonal Flu Index, "
                "7-Day Rolling Average, "
                "Local Temperature"
            )

            driver_list = [
                d.strip()
                for d in str(
                    raw_drivers
                ).split(",")
                if (
                    "facility" not in d.lower()
                    and "hospital" not in d.lower()
                )
            ]

            top_3_drivers = ", ".join(
                driver_list[:3]
            )

            manifest_display_lines.append(
                f"• **{iname}:** **{net_q}** units "
                f"*(Model-Level Important Features: "
                f"{top_3_drivers})*"
            )

        active_hospital = st.session_state.get(
            "hospital",
            "Main General Hospital"
        )

        active_username = st.session_state.get(
            "username",
            "Pharmacist"
        )

        order_status_val = st.session_state.get(
            "order_status",
            "Pending Approval"
        )

        st.write(
            "**Order ID:** PO-2026-8841"
        )

        st.write(
            "**Supplier:** Gulf Medical Supplies Co."
        )

        st.write(
            f"**Target Facility:** {active_hospital}"
        )

        st.write(
            f"**Status:** {order_status_val}"
        )

        st.markdown(
            "**Net Procurement Quantities "
            "(Day +7 Demand + 20% Forecast Buffer "
            "- On-Hand Inventory):**"
        )

        for line in manifest_display_lines:
            st.markdown(line)

        email_manifest_str = "\n".join(
            [
                f"- {k}: {v} units"
                for k, v in quantities_summary.items()
            ]
        )

        email_draft = st.text_area(
            "Supplier Communication Draft",
            value=f"""
Subject: URGENT - Automated Restock Dispatch Request for {active_hospital}

Dear Gulf Medical Supplies Dispatch Team,

In accordance with StockPulse AI automated inventory forecasting protocols, facility {active_hospital} requires an emergency comprehensive resupply batch across all tracked inventory items driven by predicted Day +7 consumption metrics.

Requested Net Resupply Manifest:
{email_manifest_str}

Please confirm dispatch timeline and certificate of compliance.

Sincerely,
{active_username}
Chief Pharmacist / Supply Chain Operations
""",
            height=200
        )


        col_act1, col_act2 = st.columns(2)


        with col_act1:

            if st.button(
                "Approve Order",
                use_container_width=True,
                type="primary"
            ):

                st.session_state.order_status = (
                    "Approved"
                )

                st.success(
                    "Order approved successfully. "
                    "Dispatch status updated in the "
                    "StockPulse procurement workflow."
                )


        with col_act2:

            reject_clicked = st.button(
                "Reject / Flag Order",
                use_container_width=True
            )


        if reject_clicked:

            st.session_state.order_status = (
                "Rejected"
            )


        if (
            st.session_state.get(
                "order_status"
            )
            == "Rejected"
        ):

            st.markdown(
                """
                <div class='alert-red'>
                    <b>Order Status: REJECTED</b>.
                    Logged in Audit History.
                </div>
                """,
                unsafe_allow_html=True
            )

            rejection_reason = st.text_input(
                "Enter Pharmacist Rejection Reason:",
                value=(
                    "Stock on hand exceeds expected "
                    "surge threshold for active catalog."
                ),
                key="rej_reason_input"
            )

            if st.button(
                "Save Rejection Record to Audit Trail",
                key="save_rej_btn"
            ):

                log_entry = {
                    "timestamp":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "order_id":
                        "PO-2026-8841",

                    "facility":
                        active_hospital,

                    "pharmacist":
                        active_username,

                    "quantities":
                        str(quantities_summary),

                    "reason":
                        rejection_reason
                }

                if (
                    "rejection_logs"
                    not in st.session_state
                ):
                    st.session_state.rejection_logs = []

                st.session_state.rejection_logs.append(
                    log_entry
                )

                st.success(
                    "Rejection logged successfully "
                    "to compliance audit history."
                )


    # ==========================================
    # RAG COMPLIANCE
    # ==========================================

    with col_o2:

        st.markdown(
            "### 🛡️ RAG Compliance & Policy Verification"
        )

        st.markdown(
            "Cross-check this purchase order against "
            "official hospital procurement manuals, "
            "storage requirements, and regulatory "
            "compliance rules."
        )

        active_items_str = ", ".join(
            df_f["Item_Name"].tolist()
        )

        default_prompt = (
        "Evaluate the following proposed procurement order against the uploaded hospital "
        "procurement and compliance knowledge base.\n\n"
        f"Facility: {active_hospital}\n"
        f"Proposed order:\n{email_manifest_str}\n\n"
        "Briefly identify:\n"
        "- Any documented quantity or procurement thresholds that apply.\n"
        "- Any approval or review requirements triggered by the proposed quantities.\n"
        "- Any documented storage, handling, or documentation requirements.\n"
        "- Any part of the proposed order that requires additional attention according to the KB.\n\n"
        "Base the assessment ONLY on the uploaded knowledge base. "
        "If the KB does not specify a relevant rule, state that clearly. "
        "Keep the response concise."
        )



        policy_query_input = st.text_area(
            "RAG Compliance Prompt",
            value=default_prompt,
            height=120,
            key="rag_compliance_prompt_input",
            help=(
                "Modify this query to double-check "
                "specific policy items against uploaded documents."
            )
        )

        if st.button(
            "Double-Check Order Policies (RAG)",
            use_container_width=True,
            key="rag_double_check_btn"
        ):
            with st.spinner("Reviewing uploaded compliance guidelines..."):

                active_rag = (
                    st.session_state.get("rag_chain")
                    or (rag_chain if "rag_chain" in globals() else None)
                )

                if active_rag is not None:
                    try:
                        # Send the proposed order, including actual quantities, to the RAG
                        response = active_rag.invoke({
                            "input": policy_query_input
                        })

                        # Get the generated compliance assessment
                        ans = response.get(
                            "answer",
                            "No answer returned."
                        )

                        # Get the actual documents retrieved from the knowledge base
                        source_docs = response.get(
                            "context",
                            []
                        )

                        # RAG COMPLIANCE ANSWER
                        st.markdown("### 🛡️ Policy & Compliance Summary")

                        st.markdown(
                            f"""
                            <div class='rag-box'>
                                {ans}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # SOURCE DOCUMENTS
                        if source_docs:
                            st.markdown("### 📚 Source Documents")

                            for i, doc in enumerate(source_docs, 1):

                                source = doc.metadata.get(
                                    "source",
                                    "Unknown document"
                                )

                                page = doc.metadata.get("page")

                                if page is not None:
                                    st.markdown(
                                        f"**{i}. {source} — Page {page + 1}**"
                                    )
                                else:
                                    st.markdown(
                                        f"**{i}. {source}**"
                                    )

                        else:
                            st.caption(
                                "No source documents were returned by the RAG retriever."
                            )

                    except Exception as e:
                        st.error(
                            f"Error querying RAG model: {e}"
                        )

                else:
                    st.warning(
                        "RAG compliance module is not available."
                    )

    # ------REJECTION AUDIT LOGS------

    st.markdown("---")

    st.markdown(
        "### 📝 Pharmacist Rejection Audit Logs"
    )

    if (
        "rejection_logs"
        in st.session_state
        and st.session_state.rejection_logs
    ):

        df_logs = pd.DataFrame(
            st.session_state.rejection_logs
        )

        st.dataframe(
            df_logs,
            use_container_width=True
        )

    else:

        st.info(
            "No rejected orders logged in the current session."
        )