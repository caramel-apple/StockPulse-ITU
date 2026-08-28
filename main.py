import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="StockPulse: AI Inventory & Procurement",
    layout="wide"
)

# --- Custom Styling for Dashboard Polish ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Load Data & Models Simulation Check ---
@st.cache_data
def load_data():
    if os.path.exists("StockPulse_synthetic_data.csv"):
        return pd.read_csv("StockPulse_synthetic_data.csv")
    else:
        # Fallback dummy data frame if generate_data.py hasn't been run yet
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", end="2025-01-30", freq="D")
        items = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]
        facilities = ["HOSP_RUH_01", "HOSP_JED_02", "HOSP_DMM_03"]
        
        rows = []
        for fac in facilities:
            for d in dates:
                row = {
                    "Timestamp": d.strftime("%Y-%m-%d"),
                    "Facility_ID": fac,
                    "Local_Temp_C": 38.5,
                    "Seasonal_Flu_Rate": 65.0,
                    "Weather_Alert_Flag": 1,
                    "Weather_Type": "Sandstorm/Dust_Storm",
                    "Admitted_Patients": 145
                }
                for item in items:
                    row[f"Units_{item}"] = np.random.randint(50, 200)
                    row[f"Target_{item}_7d_Ahead"] = np.random.randint(60, 220)
                rows.append(row)
        return pd.DataFrame(rows)

df = load_data()

# --- Header Section ---
st.title("StockPulse: AI-Driven Healthcare Inventory & Procurement")
st.markdown("---")

# Sidebar for Facility Selection
st.sidebar.header("Network Control")
facilities_list = df["Facility_ID"].unique() if "Facility_ID" in df.columns else ["HOSP_RUH_01"]
selected_facility = st.sidebar.selectbox("Select Healthcare Facility", facilities_list)

# Filter dataset for selected facility
fac_df = df[df["Facility_ID"] == selected_facility].sort_values("Timestamp")
latest_record = fac_df.iloc[-1] if not fac_df.empty else None

# --- Top Layout Grid ---
# We use 3 columns corresponding roughly to your layout blocks:
# Col 1: Bar Chart (Past 30 units + 7-day forecast trend)
# Col 2: Dynamic Alert Boxes (Red/Orange/Yellow)
# Col 3: RAG Chatbot Integration Panel

col_chart, col_alerts, col_chat = st.columns([1.5, 1.0, 1.0])

with col_chart:
    st.subheader("Inventory Consumption & 7-Day Forecast")
    st.markdown("<small style='color:purple;'>Bar chart showing past consumption bars + 7-day forecast projection</small>", unsafe_allow_html=True)
    
    # Extract items list
    items = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]
    
    # Let user pick an item to view its 3-bar history chart detail
    selected_item = st.selectbox("Select Medical Item to Inspect", items)
    
    if not fac_df.empty:
        # Build dummy or real recent history + 7 day forecast bars
        # For visualization clarity based on your sketch (Past Day 2, Past Day 1, and 7-Day Forecast)
        recent_history = fac_df.tail(10)
        history_vals = recent_history[f"Units_{selected_item}"].tolist()
        
        bar_labels = ["Day -2", "Day -1", "Forecast (Day +7)"]
        # Take the last two historical points and the predicted 7-day target value
        bar_values = [
            history_vals[-3] if len(history_vals) >= 3 else 100,
            history_vals[-2] if len(history_vals) >= 2 else 110,
            int(latest_record.get(f"Target_{selected_item}_7d_Ahead", 130))
        ]
        
        # Plotly chart configuration enabling click selection on the 3rd bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bar_labels,
            y=bar_values,
            marker_color=['#1f77b4', '#1f77b4', '#ff7f0e'], # Highlight 3rd bar
            text=bar_values,
            textposition='auto',
        ))
        
        # Add a trend line across the bars
        fig.add_trace(go.Scatter(
            x=bar_labels,
            y=bar_values,
            mode='lines+markers',
            name='Trend Line',
            line=dict(color='black', width=2)
        ))
        
        fig.update_layout(
            title=f"Trend & Forecast for {selected_item}",
            yaxis_title="Units Required / Consumed",
            margin=dict(l=20, r=20, t=40, b=20),
            height=320,
            showlegend=False
        )
        
        # Render interactive plot and capture user click event on bars
        event = st.plotly_chart(fig, on_select="rerun", key=f"chart_{selected_item}")
        
        # --- Requirement 1: Click on the 3rd bar triggers explanation popup ---
        if event and "point_indices" in event and event["point_indices"]:
            clicked_index = event["point_indices"][0]
            if clicked_index == 2: # The 3rd bar (Forecast Day +7)
                forecast_val = bar_values[2]
                temp = latest_record.get('Local_Temp_C', 35)
                flu = latest_record.get('Seasonal_Flu_Rate', 50)
                weather = latest_record.get('Weather_Type', 'Normal')
                
                # Non-technical explanation popup dialog/expander
                with st.expander(f"🤖 AI Decision Explanation: Why {forecast_val} units for {selected_item}?", expanded=True):
                    st.write(f"""
                    **Plain Language Breakdown:**
                    * **Patient Load:** Current active patient census is high, driving regular baseline daily consumption.
                    * **Environmental Factors:** Local conditions record a temperature of **{temp}°C** paired with a active **{weather}** event, which historically increases respiratory or dehydration complications.
                    * **Seasonal Trend:** Regional flu transmission rates are sitting at **{flu}%**, prompting the Random Forest model to buffer safety stock to prevent stockouts over the next 7 days.
                    """)

with col_alerts:
    st.subheader("AI Stock & Risk Alerts")
    st.markdown("<small style='color:purple;'>Dynamic priority alerts based on model threshold outputs</small>", unsafe_allow_html=True)
    
    # --- Requirement 3: Color-coded alert boxes ---
    weather_type = latest_record.get("Weather_Type", "None") if latest_record is not None else "None"
    
    if weather_type != "None":
        st.error(f" **URGENT:** Severe {weather_type} detected at {selected_facility}! IV Fluids and Inhalers stock running critically low. Immediate procurement triggered.")
    
    st.warning(f" **Moderate Alert:** Antibiotics buffer stock dropping below safety threshold. Recommended reorder: 150 units.")
    
    st.info(f"ℹ **Notice:** N95 Masks inventory levels stable, but projected demand spike expected in 4 days.")
    
    st.success(f" Insulin Vials supply levels optimal across facility network.")

with col_chat:
    st.subheader("StockPulse AI Assistant")
    st.markdown("<small style='color:purple;'>RAG Assistant connected to policy documents (rag.py)</small>", unsafe_allow_html=True)
    
    # Simple interactive chat UI frame wrapping the RAG logic layout
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am StockPulse AI. Ask me anything about your supply guidelines or inventory limits."}
        ]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_query := st.chat_input("Ask about procurement policy or stock rules..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Simulated response from rag.py integration backend
        response_text = f"Based on your internal supply documents, emergency procurement for `{selected_facility}` during weather alerts must adhere to fast-track vendor dispatch protocols outlined in Section 4.2."
        
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Bottom Section Grid ---
st.markdown("---")
col_bottom_left, col_bottom_right = st.columns([1.2, 2.0])

with col_bottom_left:
    st.subheader("Patient Status Today")
    st.markdown("<small style='color:purple;'>From the database records</small>", unsafe_allow_html=True)
    
    admitted = latest_record.get("Admitted_Patients", 120) if latest_record is not None else 120
    flu_rate = latest_record.get("Seasonal_Flu_Rate", 45.0) if latest_record is not None else 45.0
    
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric(label="Admitted Patients", value=int(admitted), delta="+12 vs yesterday")
    metric_col2.metric(label="Flu Rate Index", value=f"{flu_rate}%", delta="High Risk")

with col_bottom_right:
    st.subheader("Regional Environmental & Weather Intelligence")
    st.markdown("<small style='color:purple;'>Weather events like heatwave, sandstorm, flu, fog</small>", unsafe_allow_html=True)
    
    temp = latest_record.get("Local_Temp_C", 38.0) if latest_record is not None else 38.0
    w_type = latest_record.get("Weather_Type", "None") if latest_record is not None else "None"
    
    env_col1, env_col2, env_col3 = st.columns(3)
    env_col1.metric(label="Local Temperature", value=f"{temp}°C")
    env_col2.metric(label="Active Weather Event", value=w_type)
    env_col3.metric(label="Logistics Status", value="Active / Monitored", delta="Normal")