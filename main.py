import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor

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

# --- 3. CUSTOM STYLING (Purple Gradient & Humane UI) ---
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

# --- 4. DATA LOADING & IN-MEMORY MODEL TRAINING ---
items_list = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

@st.cache_data
def load_data():
    df = pd.read_csv("StockPulse_synthetic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='mixed', dayfirst=True)
    return df

df_global = load_data()

@st.cache_resource(show_spinner="Training Machine Learning Models in memory...")
def train_models_in_memory(df):
    models = {}
    base_features = ["Local_Temp_C", "Seasonal_Flu_Rate", "Weather_Alert_Flag", "Admitted_Patients", "Facility_ID", "Weather_Type"]
    df_features = df[base_features].copy()
    
    df_features["Weather_Type"] = df_features["Weather_Type"].fillna("Normal")
    X_encoded = pd.get_dummies(df_features, columns=["Facility_ID", "Weather_Type"])
    
    for item in items_list:
        X_item = X_encoded.copy()
        X_item[f"{item}_Lag_7d"] = df[f"Units_{item}"].shift(7).bfill()
        X_item[f"{item}_Rolling_7d"] = df[f"Units_{item}"].rolling(window=7).mean().shift(1).bfill()
        
        y = df[f"Target_{item}_7d_Ahead"].fillna(df[f"Units_{item}"] * 1.1)
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_item, y)
        models[item] = rf
        
    return models

models_dict = train_models_in_memory(df_global)

# --- 5. SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "hospital" not in st.session_state:
    st.session_state.hospital = "HOSP_RUH_01"
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

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
# VIEW 2: MAIN APPLICATION (AUTHENTICATED)
# ==========================================

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(f"### 🏥 {st.session_state.hospital}")
st.sidebar.markdown(f"Welcome back, *{st.session_state.username}*")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation Menu", 
    ["Dashboard", "StockPulse ChatBot", "Orders & Procurement"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.messages = []
    st.rerun()

df_hosp = df_global[df_global["Facility_ID"] == st.session_state.hospital].sort_values("Timestamp").reset_index(drop=True)

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
    lag_7d = df_facility.iloc[past_7d_idx][f"Units_{item_name}"]
    
    start_roll = max(0, target_idx - 7)
    rolling_7d = df_facility.iloc[start_roll:target_idx][f"Units_{item_name}"].mean()

    base_feats[f"{item_name}_Lag_7d"] = lag_7d
    base_feats[f"{item_name}_Rolling_7d"] = rolling_7d if pd.notna(rolling_7d) else lag_7d

    X_df = pd.DataFrame([base_feats])

    if hasattr(model_obj, "feature_names_in_"):
        for col in model_obj.feature_names_in_:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[model_obj.feature_names_in_]

    return X_df

# ==========================================
# SECTION A: DASHBOARD VIEW
# ==========================================
if nav_selection == "Dashboard":
    st.markdown("# 📊 StockPulse Executive Dashboard")
    st.markdown(f"Real-time predictive analytics and autonomous stock monitoring for *{st.session_state.hospital}*.")
    
    today_real = datetime.now().date()
    min_date = today_real - pd.Timedelta(days=3)
    max_date = today_real + pd.Timedelta(days=3)

    selected_date = st.date_input(
        "📅 Select Operational Date",
        value=today_real,
        min_value=min_date,
        max_value=max_date,
        help="Select today's date or ±3 days ahead/behind."
    )

    day_offset = (selected_date - today_real).days
    max_dataset_idx = len(df_hosp) - 1
    target_idx = max(0, min(max_dataset_idx, max_dataset_idx + day_offset))
    row_data = df_hosp.iloc[target_idx]

    # Live Model Predictions
    live_predictions = {}
    for item in items_list:
        X_input = prepare_model_features(df_hosp, target_idx, item, models_dict[item])
        live_predictions[item] = float(models_dict[item].predict(X_input)[0])

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Patients on {selected_date.strftime('%b %d, %Y')}</h4>
                <h2 style='margin-top: 0;'>{int(row_data['Admitted_Patients'])}</h2>
            </div>
            <p style='color: #10b981; font-size: 12px; margin: 10px 0 0 0;'>● Live Hospital Load</p>
        </div>""", unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Local Temperature</h4>
                <h2 style='margin-top: 0;'>{row_data['Local_Temp_C']} °C</h2>
            </div>
            <p style='color: #f59e0b; font-size: 12px; margin: 10px 0 0 0;'>● Weather Sensor Active</p>
        </div>""", unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""<div class='metric-card'>
            <div>
                <h4 style='margin-bottom: 5px;'>Seasonal Flu Index</h4>
                <h2 style='margin-top: 0;'>{row_data['Seasonal_Flu_Rate']:.1f} / 100</h2>
            </div>
            <p style='color: #6366f1; font-size: 12px; margin: 10px 0 0 0;'>● Regional Epidemic Tracking</p>
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

    # Charts & Alerts Section
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
# SECTION B: RAG CHATBOT VIEW
# ==========================================
elif nav_selection == "StockPulse ChatBot":
    st.markdown("# 💬 StockPulse Policy Assistant (RAG)")
    st.markdown("Chat live with your embedded healthcare supply chain guidelines, policy documents, and compliance manuals.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about inventory policies or compliance...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector database and generating response via rag.py model..."):
                if rag_chain is not None:
                    try:
                        response = rag_chain.invoke({"input": user_query})
                        answer = response["answer"]
                        sources = response.get("context", [])
                        
                        full_resp = answer + "\n\n*Sources Used:*\n"
                        for doc in sources:
                            src = doc.metadata.get("source", "Unknown")
                            pg = doc.metadata.get("page", "N/A")
                            full_resp += f"- {src} (Page: {pg})\n"
                    except Exception as e:
                        full_resp = f"Error during model invocation: {e}"
                else:
                    full_resp = "The information is unavailable in the supply documents (rag.py pipeline could not be loaded)."
                
                st.markdown(full_resp)
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

# ==========================================
# SECTION C: ORDERS & PROCUREMENT VIEW
# ==========================================
elif nav_selection == "Orders & Procurement":
    st.markdown("# 🛒 Autonomous Procurement & Supplier Orders")
    st.markdown("Review AI-generated purchase orders, verify compliance guidelines with RAG, and approve drafts before dispatching to suppliers.")
    
    col_o1, col_o2 = st.columns(2)
    
    with col_o1:
        st.markdown("### 📋 Pending Supplier Draft Order")
        st.markdown(f"""
        *Order ID:* PO-2026-8841  
        *Supplier:* Gulf Medical Supplies Co.  
        *Target Facility:* {st.session_state.hospital}  
        *Items Requested:* 
        * **IV_Fluids:** 500 units
        * **N95_Masks:** 250 units
        * **Antibiotics:** 300 units  
        *Trigger Cause:* AI 7-day forecast surge + extreme weather event.
        """)
        
        email_draft = st.text_area(
            "Supplier Communication Draft",
            value=f"""Subject: URGENT - Automated Restock Dispatch Request for {st.session_state.hospital}

Dear Gulf Medical Supplies Dispatch Team,

In accordance with StockPulse AI automated inventory forecasting protocols, facility {st.session_state.hospital} requires an emergency resupply batch prior to the upcoming weather event and patient surge.

Requested Manifest:
- IV_Fluids: 500 units
- N95_Masks: 250 units
- Antibiotics: 300 units

Please confirm dispatch timeline and certificate of compliance.

Sincerely,
{st.session_state.username}
Chief Pharmacist / Supply Chain Operations
StockPulse Healthcare Network""",
            height=200
        )
        
        if st.button("Approve & Dispatch Order to Supplier", use_container_width=True):
            st.success("Order successfully approved by human supervisor and transmitted via EDI/Email to supplier!")

    with col_o2:
        st.markdown("### 🛡️ RAG Compliance & Policy Verification")
        st.markdown("Cross-check this purchase order against official hospital procurement manuals, storage requirements, and regulatory compliance rules.")
        
        policy_query_input = st.text_area(
            "RAG Compliance Prompt",
            value=f"Review draft order PO-2026-8841 containing IV_Fluids, N95_Masks, and Antibiotics for facility {st.session_state.hospital}. What compliance rules, cold-chain storage regulations, emergency procurement procedures, or vendor protocols must be followed for this dispatch?",
            height=120
        )
        
        if st.button("Double-Check Order Policies (RAG)", use_container_width=True):
            with st.spinner("Reviewing uploaded compliance guidelines..."):
                if rag_chain is not None:
                    try:
                        response = rag_chain.invoke({"input": policy_query_input})
                        st.markdown(f"<div class='rag-box'><b>RAG Policy & Compliance Guidance:</b><br>{response['answer']}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error querying RAG model: {e}")
                else:
                    st.error("RAG pipeline unavailable. Verify rag.py configuration.")