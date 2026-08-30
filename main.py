import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor

# Hardcoded Baseline Inventory on Hand per Item
HARDCODED_INVENTORY = {
    "N95_Masks": 450,       # High stock baseline
    "IV_Fluids": 120,       # Moderate stock
    "Antibiotics": 80,      # Low stock
    "Inhalers": 40,         # Critical low stock
    "Insulin_Vials": 110,   # Moderate stock
    "Painkillers": 200      # Healthy stock
}

# Safety buffer percentage per item (20% buffer over forecast)
SAFETY_BUFFER_PCT = 0.20

# --- 1. IMPORT RAG PIPELINE FROM rag.py ---
try:
    from rag import rag_chain
except Exception as e:
    rag_chain = None
    print(f"[!] Warning: Could not load rag_chain directly from rag.py: {e}")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockPulse: AI Inventory Forecasting & Procurement",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 50%, #ddd6fe 100%) !important;
    }
    .stSidebar {
        background-color: #faf5ff !important;
        border-right: 1px solid #e9d5ff;
    }
    h1, h2, h3 {
        color: #26144e !important;
        font-family: 'Playfair Display', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(91, 33, 182, 0.1);
        border-left: 5px solid #7c3aed;
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
    .alert-red {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 12px;
        border-radius: 8px;
        color: #991b1b;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .alert-orange {
        background-color: #ffedd5;
        border-left: 5px solid #f97316;
        padding: 12px;
        border-radius: 8px;
        color: #9a3412;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .alert-yellow {
        background-color: #fef9c3;
        border-left: 5px solid #eab308;
        padding: 12px;
        border-radius: 8px;
        color: #854d0e;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .rag-box {
        background-color: #ffffff;
        border-left: 5px solid #6366f1;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADING & MODEL TRAINING ---
items_list = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

@st.cache_data
def load_data():
    df = pd.read_csv("StockPulse_synthetic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='mixed', dayfirst=True)
    df = df.sort_values(["Facility_ID", "Timestamp"]).reset_index(drop=True)
    return df

df_global = load_data()

@st.cache_resource(show_spinner="Training Multivariate Forecasting Models...")
def train_models_in_memory(df):
    models = {}
    df_proc = df.copy()
    df_proc["Weather_Type"] = df_proc["Weather_Type"].fillna("Normal")
    
    for item in items_list:
        df_proc[f"{item}_Lag_7d"] = df_proc.groupby("Facility_ID")[f"Units_{item}"].shift(7).bfill()
        df_proc[f"{item}_Rolling_7d"] = df_proc.groupby("Facility_ID")[f"Units_{item}"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean().shift(1)
        ).bfill()

    df_encoded = pd.get_dummies(df_proc, columns=["Facility_ID", "Weather_Type"])

    base_features = ["Local_Temp_C", "Seasonal_Flu_Rate", "Weather_Alert_Flag", "Admitted_Patients"]
    facility_cols = [c for c in df_encoded.columns if c.startswith("Facility_ID_")]
    weather_cols = [c for c in df_encoded.columns if c.startswith("Weather_Type_")]

    for item in items_list:
        item_features = base_features + facility_cols + weather_cols + [f"{item}_Lag_7d", f"{item}_Rolling_7d"]
        X_item = df_encoded[item_features]
        
        target_col = f"Target_{item}_7d_Ahead"
        y = df_encoded[target_col].fillna(df_encoded[f"Units_{item}"] * 1.1) if target_col in df_encoded.columns else df_encoded[f"Units_{item}"] * 1.1

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_item, y)
        models[item] = rf
        
    return models

models_dict = train_models_in_memory(df_global)

def prepare_model_features(df_facility, target_idx, item_name, model_obj):
    current_row = df_facility.iloc[target_idx]
    
    base_feats = {
        "Local_Temp_C": current_row["Local_Temp_C"],
        "Seasonal_Flu_Rate": current_row["Seasonal_Flu_Rate"],
        "Weather_Alert_Flag": current_row["Weather_Alert_Flag"],
        "Admitted_Patients": current_row["Admitted_Patients"]
    }
    
    for fac in ["HOSP_RUH_01", "HOSP_JED_02", "HOSP_DMM_03"]:
        base_feats[f"Facility_ID_{fac}"] = 1 if current_row["Facility_ID"] == fac else 0
        
    w_val = current_row["Weather_Type"] if pd.notna(current_row["Weather_Type"]) else "Normal"
    base_feats["Weather_Type_Heatwave"] = 1 if w_val == "Heatwave" else 0
    base_feats["Weather_Type_Sandstorm/Dust_Storm"] = 1 if w_val in ["Sandstorm", "Dust_Storm", "Sandstorm/Dust_Storm"] else 0
    base_feats["Weather_Type_Normal"] = 1 if w_val in ["None", "Normal", ""] else 0

    past_7d_idx = max(0, target_idx - 7)
    base_feats[f"{item_name}_Lag_7d"] = df_facility.iloc[past_7d_idx][f"Units_{item_name}"]
    
    start_roll = max(0, target_idx - 7)
    rolling_val = df_facility.iloc[start_roll:target_idx][f"Units_{item_name}"].mean()
    base_feats[f"{item_name}_Rolling_7d"] = rolling_val if pd.notna(rolling_val) else base_feats[f"{item_name}_Lag_7d"]

    X_df = pd.DataFrame([base_feats])

    if hasattr(model_obj, "feature_names_in_"):
        for col in model_obj.feature_names_in_:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[model_obj.feature_names_in_]

    return X_df

# --- 5. SESSION STATE ---
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

# ==========================================
# VIEW 1: LOGIN PAGE
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>💊 StockPulse AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280;'>Standardized AI-Driven Inventory Forecasting & Procurement</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Secure Staff Portal Login")
            hospital_choice = st.selectbox(
                "Choose Healthcare Facility", 
                ["HOSP_RUH_01 (Riyadh Central)", "HOSP_JED_02 (Jeddah General)", "HOSP_DMM_03 (Dammam Medical City)"]
            )
            username = st.text_input("Username", placeholder="e.g., Dr. Sarah")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            submit_btn = st.form_submit_button("Access Portal 🚀", use_container_width=True)
            if submit_btn:
                st.session_state.logged_in = True
                st.session_state.hospital = hospital_choice.split(" ")[0]
                st.session_state.username = username if username else "Chief Pharmacist"
                st.rerun()
    st.stop()

# ==========================================
# VIEW 2: MAIN APPLICATION
# ==========================================

st.sidebar.markdown(f"### 🏥 {st.session_state.hospital}")
st.sidebar.markdown(f"Welcome back, *{st.session_state.username}*")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation Menu", 
    ["Dashboard", "Orders & Procurement"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.messages = []
    st.rerun()

df_hosp = df_global[df_global["Facility_ID"] == st.session_state.hospital].sort_values("Timestamp").reset_index(drop=True)

def generate_simulated_row(target_dt, base_r):
    facility_id = base_r["Facility_ID"]
    day_of_year = target_dt.timetuple().tm_yday
    month = target_dt.month
    day_of_week = target_dt.weekday()
    
    date_int = target_dt.year * 10000 + target_dt.month * 100 + target_dt.day
    fac_code = sum(ord(c) for c in facility_id)
    rng = np.random.RandomState(seed=date_int + fac_code)

    base_temp = 31.0 + 11.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    temp_noise = rng.normal(0, 1.8)
    sim_temp = float(np.clip(base_temp + temp_noise, 12.0, 50.0))

    is_summer = month in [6, 7, 8, 9]
    is_spring = month in [3, 4, 5]
    
    sim_weather = "None"
    temp_boost = 0.0
    
    if is_summer and sim_temp > 38.0 and rng.rand() < 0.15:
        sim_weather = "Heatwave"
        temp_boost = rng.uniform(4.0, 7.0)
    elif is_spring and rng.rand() < 0.12:
        sim_weather = "Sandstorm/Dust_Storm"

    sim_temp = round(float(np.clip(sim_temp + temp_boost, 12.0, 52.0)), 1)
    sim_alert = 1 if sim_weather != "None" else 0

    flu_base = 65.0 + 30.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    flu_noise = rng.normal(0, 4.0)
    sim_flu = round(float(np.clip(flu_base + flu_noise, 5.0, 100.0)), 2)

    facility_baselines = {"HOSP_RUH_01": 220, "HOSP_JED_02": 150, "HOSP_DMM_03": 95}
    base_capacity = facility_baselines.get(facility_id, 150)

    flu_surge = (sim_flu / 100.0) * (base_capacity * 0.25)
    weather_surge = base_capacity * 0.20 if sim_weather == "Heatwave" else (base_capacity * 0.30 if sim_weather == "Sandstorm/Dust_Storm" else 0)
    
    dow_multipliers = [1.05, 1.08, 1.06, 1.02, 0.95, 0.85, 0.88]
    dow_mult = dow_multipliers[day_of_week]
    patient_noise = rng.normal(0, base_capacity * 0.05)

    total_patients = (base_capacity + flu_surge + weather_surge + patient_noise) * dow_mult
    sim_patients = int(max(20, round(total_patients)))

    sim_row = base_r.copy()
    sim_row["Local_Temp_C"] = sim_temp
    sim_row["Seasonal_Flu_Rate"] = sim_flu
    sim_row["Admitted_Patients"] = sim_patients
    sim_row["Weather_Type"] = sim_weather
    sim_row["Weather_Alert_Flag"] = sim_alert

    item_factors = {"N95_Masks": 1.8, "IV_Fluids": 2.4, "Antibiotics": 1.2, "Inhalers": 0.8, "Insulin_Vials": 0.6, "Painkillers": 2.1}

    for item, factor in item_factors.items():
        item_boost = 1.0
        if item == "Inhalers" and sim_weather == "Sandstorm/Dust_Storm":
            item_boost *= 2.2
        if item == "IV_Fluids" and sim_weather == "Heatwave":
            item_boost *= 1.8
        if item in ["N95_Masks", "Antibiotics"] and sim_flu > 70:
            item_boost *= 1.5

        item_units = sim_patients * factor * item_boost * rng.uniform(0.9, 1.1)
        sim_row[f"Units_{item}"] = int(max(5, round(item_units)))

    return sim_row

# ==========================================
# SECTION A: DASHBOARD VIEW
# ==========================================
if nav_selection == "Dashboard":
    st.markdown("# 📊 StockPulse Executive Dashboard")
    st.markdown(f"Real-time predictive analytics and autonomous stock monitoring for *{st.session_state.hospital}*.")
    
    min_date = datetime(2026, 1, 1).date()
    max_date = datetime(2026, 12, 31).date()
    today_real = datetime.now().date()
    default_date = max(min_date, min(today_real, max_date))

    selected_date = st.date_input(
        "📅 Select Operational Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help="Select any target operational date in 2026 to generate predictive AI forecasts."
    )

    latest_idx = len(df_hosp) - 1
    base_row = df_hosp.iloc[latest_idx]

    simulated_target_row = generate_simulated_row(selected_date, base_row)

    df_simulated_context = df_hosp.copy()
    df_simulated_context.loc[len(df_simulated_context)] = simulated_target_row
    sim_idx = len(df_simulated_context) - 1
    row_data = df_simulated_context.iloc[sim_idx]

    live_predictions = {}
    predicted_orders = {}
    net_reorder_qty = {}
    current_inventory = {}
    safety_stock = {}
    item_feature_drivers = {}

    for item in items_list:
        X_input = prepare_model_features(df_simulated_context, sim_idx, item, models_dict[item])
        pred_val = float(models_dict[item].predict(X_input)[0])
        pred_demand = int(np.ceil(pred_val))

        live_predictions[item] = pred_val
        predicted_orders[item] = pred_demand

        curr_stock = HARDCODED_INVENTORY.get(item, 100)
        current_inventory[item] = curr_stock

        buffer_qty = max(10, int(np.ceil(pred_demand * SAFETY_BUFFER_PCT)))
        safety_stock[item] = buffer_qty

        # Net reorder formula: (Demand + Buffer) - Stock
        net_order = max(0, (pred_demand + buffer_qty) - curr_stock)
        net_reorder_qty[item] = net_order

        # STRICT FEATURE DRIVER FILTERING
        model = models_dict[item]
        if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_
            sorted_indices = np.argsort(importances)[::-1]

            friendly_names = {
                "Admitted_Patients": "Admitted Patients",
                "Seasonal_Flu_Rate": "Seasonal Flu Index",
                "Local_Temp_C": "Local Temperature",
                "Weather_Alert_Flag": "Extreme Weather Alert",
                f"{item}_Lag_7d": "7-Day Historical Demand",
                f"{item}_Rolling_7d": "7-Day Rolling Average"
            }

            clean_drivers = []
            for idx in sorted_indices:
                feat = feature_names[idx]
                # Exclude dummy columns (Facility_ID & Weather_Type)
                if "Facility" in feat or "Weather_Type" in feat:
                    continue
                readable_name = friendly_names.get(feat, feat.replace("_", " "))
                clean_drivers.append(readable_name)
                if len(clean_drivers) == 3:
                    break

            item_feature_drivers[item] = ", ".join(clean_drivers)
        else:
            item_feature_drivers[item] = "Admitted Patients, Seasonal Flu Index, Local Temperature"

    st.session_state.predicted_orders = predicted_orders
    st.session_state.current_inventory = current_inventory
    st.session_state.safety_stock = safety_stock
    st.session_state.net_reorder_qty = net_reorder_qty
    st.session_state.item_feature_drivers = item_feature_drivers

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Patients on {selected_date.strftime('%b %d, %Y')}</h4>
                <h2 style='margin-top: 0;'>{int(row_data['Admitted_Patients'])}</h2>
            </div>
            <p style='color: #10b981; font-size: 12px; margin: 10px 0 0 0;'>● Projected Hospital Load</p>
        </div>""", unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Local Temperature</h4>
                <h2 style='margin-top: 0;'>{row_data['Local_Temp_C']} °C</h2>
            </div>
            <p style='color: #f59e0b; font-size: 12px; margin: 10px 0 0 0;'>● Forecast Sensor</p>
        </div>""", unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Seasonal Flu Index</h4>
                <h2 style='margin-top: 0;'>{row_data['Seasonal_Flu_Rate']:.1f} / 100</h2>
            </div>
            <p style='color: #6366f1; font-size: 12px; margin: 10px 0 0 0;'>● Epidemic Forecast</p>
        </div>""", unsafe_allow_html=True)
        
    with m4:
        weather_label = row_data['Weather_Type'] if pd.notna(row_data['Weather_Type']) else "Normal Conditions"
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Weather Status</h4>
                <h3 style='font-size: 18px; margin-top: 0;'>{weather_label}</h3>
            </div>
            <p style='color: #ef4444; font-size: 12px; margin: 10px 0 0 0;'>● Alert Flag: {int(row_data['Weather_Alert_Flag'])}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts & Alerts
    chart_col, alert_col = st.columns([2, 1])

    with chart_col:
        st.markdown("### 📈 Demand Trend & Live AI Prediction")
        selected_product = st.selectbox("Select Medical Item to Inspect", items_list)
        
        day_chosen = row_data[f"Units_{selected_product}"]
        predicted_val = live_predictions[selected_product]
        
        x_vals = [f"Selected ({selected_date.strftime('%b %d')})", "AI Forecast (+7d)"]
        y_vals = [day_chosen, predicted_val]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_vals, y=y_vals,
            marker_color=['#080949', "#9b1d52"],
            text=[f"{val:.0f} units" for val in y_vals],
            textposition='auto',
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            line=dict(color='#311059', width=3),
            marker=dict(size=8),
            name='Trend Line'
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title="Units Consumed / Predicted"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with alert_col:
        st.markdown("### 🚨 Urgent AI Notifications")
        st.markdown(
            f"<div class='alert-red'><b>Critical Stock Warning:</b> IV_Fluids stock buffer is low for {selected_date.strftime('%b %d')}.</div>", 
            unsafe_allow_html=True
        )
        if pd.notna(row_data['Weather_Type']) and row_data['Weather_Type'] != "None":
            st.markdown(
                f"<div class='alert-orange'><b>Weather Alert:</b> {row_data['Weather_Type']} active. Inhalers & N95 demand surging.</div>", 
                unsafe_allow_html=True
            )
        st.markdown(
            f"<div class='alert-yellow'><b>Model Recommendation:</b> Flu index at {row_data['Seasonal_Flu_Rate']:.1f}. Buffer recommended.</div>", 
            unsafe_allow_html=True
        )

# ==========================================
# SECTION B: ORDERS & PROCUREMENT VIEW
# ==========================================
elif nav_selection == "Orders & Procurement":
    net_reorder = st.session_state.get("net_reorder_qty", {})
    predicted_orders = st.session_state.get("predicted_orders", {})
    item_feature_drivers = st.session_state.get("item_feature_drivers", {})
    
    st.markdown("# 🛒 Autonomous Procurement & Supplier Orders")
    st.markdown("Review AI-generated purchase orders, verify compliance guidelines with RAG, and approve or reject draft orders.")
    
    col_o1, col_o2 = st.columns(2)
    
    with col_o1:
        st.markdown("### 📋 Pending Supplier Draft Order")
        
        iv_net = net_reorder.get('IV_Fluids', 248)
        n95_net = net_reorder.get('N95_Masks', 118)
        anti_net = net_reorder.get('Antibiotics', 173)

        iv_drivers = item_feature_drivers.get("IV_Fluids", "Admitted Patients, Local Temperature, Extreme Weather Alert")
        n95_drivers = item_feature_drivers.get("N95_Masks", "Seasonal Flu Index, Admitted Patients, Extreme Weather Alert")
        anti_drivers = item_feature_drivers.get("Antibiotics", "Seasonal Flu Index, Admitted Patients, 7-Day Historical Demand")

        st.markdown(f"""
        **Order ID:** PO-2026-8841  
        **Supplier:** Gulf Medical Supplies Co.  
        **Target Facility:** {st.session_state.hospital}  
        **Status:** `{st.session_state.order_status}`

        **Net Procurement Quantities (Demand + 20% Buffer - On-Hand Inventory):**
        * **IV_Fluids:** **{iv_net}** units *(Top Drivers: {iv_drivers})*
        * **N95_Masks:** **{n95_net}** units *(Top Drivers: {n95_drivers})*
        * **Antibiotics:** **{anti_net}** units *(Top Drivers: {anti_drivers})*
        """)
        
        email_draft = st.text_area(
            "Supplier Communication Draft",
            value=f"""Subject: URGENT - Automated Restock Dispatch Request for {st.session_state.hospital}

Dear Gulf Medical Supplies Dispatch Team,

In accordance with StockPulse AI automated inventory forecasting protocols, facility {st.session_state.hospital} requires an emergency resupply batch driven by predicted metrics ({iv_drivers}).

Requested Net Resupply Manifest:
- IV_Fluids: {iv_net} units
- N95_Masks: {n95_net} units
- Antibiotics: {anti_net} units

Please confirm dispatch timeline and certificate of compliance.

Sincerely,
{st.session_state.username}
Chief Pharmacist / Supply Chain Operations
StockPulse Healthcare Network""",
            height=180
        )
        
        # APPROVE / REJECT CONTROLS
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("Approve & Dispatch Order", use_container_width=True, type="primary"):
                st.session_state.order_status = "Approved & Dispatched"
                st.success("Order approved by pharmacist and transmitted via EDI to Gulf Medical Supplies Co.!")

        with col_act2:
            reject_clicked = st.button("Reject / Flag Order", use_container_width=True)

        if reject_clicked:
            st.session_state.order_status = "Rejected"

        if st.session_state.order_status == "Rejected":
            st.markdown("<div class='alert-red'><b>Order Status: REJECTED</b>. Logged in Audit History.</div>", unsafe_allow_html=True)
            rejection_reason = st.text_input("Enter Pharmacist Rejection Reason:", value="Stock on hand exceeds expected surge threshold.")
            
            if st.button("Save Rejection Record to Audit Trail"):
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_id": "PO-2026-8841",
                    "facility": st.session_state.hospital,
                    "pharmacist": st.session_state.username,
                    "quantities": f"IV: {iv_net}, N95: {n95_net}, Anti: {anti_net}",
                    "reason": rejection_reason
                }
                st.session_state.rejection_logs.append(log_entry)
                st.success("Rejection logged successfully to compliance audit history.")

    with col_o2:
        st.markdown("### 🛡️ RAG Compliance & Policy Verification")
        st.markdown("Cross-check this purchase order against official hospital procurement manuals, storage requirements, and regulatory compliance rules.")
        
        policy_query_input = st.text_area(
            "RAG Compliance Prompt",
            value=f"What are the official compliance rules, cold-chain storage requirements, and emergency procurement protocols for dispatching IV Fluids, N95 Masks, and Antibiotics to a hospital facility?",
            height=120
        )
        
        if st.button("Double-Check Order Policies (RAG)", use_container_width=True):
            with st.spinner("Reviewing uploaded compliance guidelines..."):
                if rag_chain is not None:
                    try:
                        response = rag_chain.invoke({"input": policy_query_input})
                        ans = response.get("answer", "No answer returned.")
                        
                        st.markdown(f"<div class='rag-box'><b>RAG Policy & Compliance Guidance:</b><br>{ans}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error querying RAG model: {e}")
                else:
                    st.error("RAG pipeline unavailable. Check rag.py load status.")
    # REJECTION AUDIT TRAIL DISPLAY
    st.markdown("---")
    st.markdown("### 📝 Pharmacist Rejection Audit Logs")
    if st.session_state.rejection_logs:
        df_logs = pd.DataFrame(st.session_state.rejection_logs)
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No rejected orders logged in the current session.")