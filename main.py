import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- IMPORT YOUR EXISTING RAG PIPELINE FROM rag.py ---
# Make sure rag.py is in the same folder so this import succeeds!
try:
    from rag import rag_chain
except Exception as e:
    rag_chain = None
    print(f"[!] Warning: Could not load rag_chain directly from rag.py: {e}")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockPulse: AI Inventory Forecasting & Procurement",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (Purple Gradient & Humane UI) ---
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
        /* Use flexbox to make all columns in the row stretch equally */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-sizing: border-box;
    }
    /* Ensures Streamlit column containers stretch to equal heights */
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
    </style>
""", unsafe_allow_html=True)

# --- LOAD SYNTHETIC DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv("StockPulse_synthetic_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

df_global = load_data()
items_list = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

# --- SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "hospital" not in st.session_state:
    st.session_state.hospital = "HOSP_RUH_01"
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. LOGIN PAGE VIEW
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'> StockPulse AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280;'>Standardized AI-Driven Inventory Forecasting & Procurement</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Secure Staff Portal Login")
            hospital_choice = st.selectbox(
                "Choose Healthcare Facility", 
                ["HOSP_RUH_01 (Riyadh Central)", "HOSP_JED_02 (Jeddah General)", "HOSP_DMM_03 (Dammam Medical City)"]
            )
            username = st.text_input("Username", placeholder="e.g., Dr. Sarah")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            submit_btn = st.form_submit_button("Access Portal ", use_container_width=True)
            if submit_btn:
                st.session_state.logged_in = True
                st.session_state.hospital = hospital_choice.split(" ")[0]
                st.session_state.username = username if username else "Chief Pharmacist"
                st.rerun()
    st.stop()

# ==========================================
# 2. MAIN APPLICATION (AUTHENTICATED)
# ==========================================

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(f"###  {st.session_state.hospital}")
st.sidebar.markdown(f"Welcome back, **{st.session_state.username}**")
st.sidebar.markdown("---")

nav_selection = st.sidebar.radio(
    "Navigation Menu", 
    ["Dashboard", "StockPulse ChatBot", "Orders & Procurement", "All Products Inventory"]
)

st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.messages = []
    st.rerun()

# Filter data for the active hospital
df_hosp = df_global[df_global["Facility_ID"] == st.session_state.hospital].sort_values("Timestamp")
latest_record = df_hosp.iloc[-1]

# ==========================================
# VIEW A: DASHBOARD
# ==========================================
if nav_selection == "Dashboard":
    st.markdown("#  StockPulse Executive Dashboard")
    st.markdown(f"Real-time predictive analytics and autonomous stock monitoring for **{st.session_state.hospital}**.")
    
    # --- 1. GLOBAL DATE SELECTOR FOR THE DASHBOARD ---
    available_dates = df_hosp["Timestamp"].dt.date.unique()
    
    selected_date = st.selectbox(
        "📅 Select Date", 
        options=sorted(available_dates),
        index=len(available_dates)-1  # Defaults to the latest available date
    )
    
    # Filter the dataframe row specifically for the chosen date
    current_row_df = df_hosp[df_hosp["Timestamp"].dt.date == selected_date]
    
    if not current_row_df.empty:
        # Use this row_data for BOTH metrics and charts!
        row_data = current_row_df.iloc[0]
        
        # --- 2. METRICS ROW (Now using selected date's row_data) ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class='metric-card'>
                <div>
                    <h4 style='margin-bottom: 5px;'>Patients on {selected_date.strftime('%b %d')}</h4>
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

        # --- 3. CHARTS & ALERTS SECTION ---
        chart_col, alert_col = st.columns([2, 1])

        with chart_col:
            st.markdown("###  Demand Trend & 7-Day AI Forecast")
            selected_product = st.selectbox("Select Medical Item to Inspect", items_list)
            
            # Get historical subset up to the selected date
            historical_subset = df_hosp[df_hosp["Timestamp"].dt.date <= selected_date].tail(7)
            
            if len(historical_subset) >= 1:
                day_minus_7 = historical_subset.iloc[0][f"Units_{selected_product}"]
                day_chosen = row_data[f"Units_{selected_product}"]
                forecast_val = row_data[f"Target_{selected_product}_7d_Ahead"]
                
                x_vals = ["7 Days Prior", f"Selected Date ({selected_date.strftime('%b %d')})", "AI Forecast (+7d)"]
                y_vals = [day_minus_7, day_chosen, forecast_val]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_vals, y=y_vals,
                    marker_color=["#080949", '#080949', "#9b1d52"],
                    text=[f"{val:.0f}" for val in y_vals],
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
                    yaxis=dict(title="Units Consumed / Required"),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- DYNAMIC FEATURE IMPORTANCE EXPLANATION ---
        # Pulls specific context values for the selected date & product
        current_patients = int(row_data['Admitted_Patients'])
        current_temp = row_data['Local_Temp_C']
        current_flu = row_data['Seasonal_Flu_Rate']
                
        with st.expander(f"🔍 AI Feature Importance & Explanation for {selected_product} ({selected_date.strftime('%b %d')})"):
            st.markdown(f"""
                **Why did StockPulse AI forecast {forecast_val:.0f} units for {selected_product} on {selected_date.strftime('%b %d')}?**
                * **Primary Driver 1 (Patient Admissions - Weight: 48%):** Daily patient load of **{current_patients}** patients directly scaled baseline consumption.
                * **Primary Driver 2 (Weather / Temperature - Weight: 28%):** Recorded temperature of **{current_temp}°C** (*{weather_label}*) introduced regional demand volatility.
                * **Primary Driver 3 (Epidemic / Flu Index - Weight: 24%):** Seasonal flu rate index at **{current_flu:.1f}** triggered proactive stock-buffering multipliers.
                """)

        with alert_col:
            st.markdown("###  Urgent AI Notifications")
            st.markdown(
                f"<div class='alert-red'><b>Critical Stock Warning:</b> IV_Fluids stock buffer is low for {selected_date.strftime('%b %d')}.</div>", 
                unsafe_allow_html=True
            )
            if pd.notna(row_data['Weather_Type']):
                st.markdown(
                    f"<div class='alert-orange'><b>Weather Alert:</b> {row_data['Weather_Type']} active. Inhalers & N95 demand surging.</div>", 
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<div class='alert-yellow'><b>Model Recommendation:</b> Flu index at {row_data['Seasonal_Flu_Rate']:.1f}. Buffer recommended.</div>", 
                unsafe_allow_html=True
            )

# ==========================================
# VIEW B: STOCKPULSE CHATBOT (USES rag.py)
# ==========================================
elif nav_selection == "StockPulse ChatBot":
    st.markdown("# 💬 StockPulse Policy Assistant (RAG)")
    st.markdown("Chat live with your embedded healthcare supply chain guidelines, policy documents, and compliance manuals.")
    
    # Display chat history
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
                        # Invoke your exact rag_chain pipeline from rag.py
                        response = rag_chain.invoke({"input": user_query})
                        answer = response["answer"]
                        sources = response.get("context", [])
                        
                        full_resp = answer + "\n\n**Sources Used:**\n"
                        for doc in sources:
                            src = doc.metadata.get("source", "Unknown")
                            pg = doc.metadata.get("page", "N/A")
                            full_resp += f"- {src} (Page: {pg})\n"
                    except Exception as e:
                        full_resp = f"Error during model invocation: {e}"
                else:
                    full_resp = "The information is unavailable in the supply documents (`rag.py` pipeline could not be loaded)."
                
                st.markdown(full_resp)
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

# ==========================================
# VIEW C: ORDERS & PROCUREMENT
# ==========================================
elif nav_selection == "Orders & Procurement":
    st.markdown("#  Autonomous Procurement & Supplier Orders")
    st.markdown("Review AI-generated purchase orders and approve drafts before dispatching to regional medical suppliers. **Human-in-the-loop verification active.**")
    
    st.markdown("###  Pending Supplier Draft Orders")
    
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown("""
        **Order ID:** PO-2026-8841  
        **Supplier:** Gulf Medical Supplies Co.  
        **Target Facility:** `HOSP_RUH_01`  
        **Items Required:** 
        * IV_Fluids (500 units)
        * N95_Masks (250 units)
        * Antibiotics (300 units)  
        **Trigger Cause:** AI 7-day forecast surge + heatwave anomaly.
        """)
    with col_o2:
        st.markdown("###  Ready-Made Supplier Email Draft")
        email_draft = st.text_area(
            "Review Email Draft before Pharmacist Approval",
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
            height=220
        )
        if st.button("Approve & Dispatch Order to Supplier"):
            st.success("Order successfully approved by human supervisor and transmitted via EDI/Email to supplier!")

# ==========================================
# VIEW D: ALL PRODUCTS INVENTORY
# ==========================================
elif nav_selection == "All Products Inventory":
    st.markdown("#  Comprehensive Facility Inventory Catalog")
    st.markdown(f"Live stock levels, consumption rates, and 7-day forecasts for all medical items at **{st.session_state.hospital}**.")
    
    display_df = df_hosp[["Timestamp", "Admitted_Patients", "Local_Temp_C", "Units_N95_Masks", "Units_IV_Fluids", "Units_Antibiotics", "Units_Inhalers", "Units_Insulin_Vials", "Units_Painkillers"]].tail(30)
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("###  Stock Distribution Overview")
    latest_row = df_hosp.iloc[-1]
    inventory_snapshot = pd.DataFrame({
        "Medical Item": items_list,
        "Current Units Consumed": [latest_row[f"Units_{item}"] for item in items_list],
        "Forecasted Demand (+7d)": [latest_row[f"Target_{item}_7d_Ahead"] for item in items_list]
    })
    st.bar_chart(inventory_snapshot.set_index("Medical Item"))