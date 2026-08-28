#in terminal write "python -m pip install pandas numpy"

#import libraries
import numpy as np
import pandas as pd

#data is reproducible
np.random.seed(42)

#define the timeframe between jan 2024 - jan 2026
dates = pd.date_range(start="2024-01-01", end = "2026-01-01", freq = "D")

#define 3 simulated healthcare facilities and average daily patient load
facilities = ["HOSP_RUH_01", "HOSP_JED_02", "HOSP_DMM_03"]
facility_baselines = {
    "HOSP_RUH_01": 220,  
    "HOSP_JED_02": 150,  
    "HOSP_DMM_03": 90,   
}

#define medical supply items 
items = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

#initialize empty list to hold data
data = []

#loop through every facility, item type, day
for facility in facilities:
    base_patients = facility_baselines[facility]
    #track active weather states across days for multi-day persistence
    active_weather = "None"
    weather_days_remaining = 0
    for date in dates:
        #extract month, year to calculate season
        month = date.month
        #define seasons 
        is_winter = month in [11, 12, 1, 2]
        is_summer = month in [6,7,8,9]

        #multi_day weather persistence
        if weather_days_remaining > 0:
            #weather event continues from previous days
            weather_days_remaining -=1
        else:
            #check if new weather event triggers
            new_event_roll = np.random.rand()
            if new_event_roll < 0.04:
                #spring (prime dust storm season)
                if month in [3,4,5]:
                    if np.random.rand() < 0.85:
                        active_weather = "Sandstorm/Dust_Storm"
                        weather_days_remaining = np.random.randint(1,3)
                    else:
                        active_weather = "Heatwave"
                        weather_days_remaining = np.random.randint(2,4)
                if is_summer:
                    if np.random.rand() < 0.75:
                        active_weather = "Heatwave"
                        weather_days_remaining = np.random.randint(3,6)
                    else:
                        active_weather = "Sandstorm/Dust_Storm"
                        weather_days_remaining = np.random.randint(1,3)
                else: #autumn/winter
                    if np.random.rand() < 0.35:
                        active_weather = "Sandstorm/Dust_Storm"
                        weather_days_remaining = np.random.randint(1,2)
                    else:
                        active_weather = "None"
            else:
                active_weather = "None"

        #flags based on active weather state
        weather_alert = 1 if active_weather != "None" else 0
        weather_type = active_weather

        #simulate regional flu rates (higher in winter)
        raw_flu = np.random.normal(50,10) + (25 if is_winter else 0)
        flu_rate = np.clip(raw_flu, 0, 100)

        #temperature curve
        day_of_year = date.dayofyear
        #centers the  sine wave to 21 high in Jan ans 43 high in the summer
        annual_temp_cycle = 32.5 + 10.5 * np.sin(2 * np.pi * (day_of_year - 110) / 365)
        #daily high variance with 16-48 as boundaries, conditional temp boos during heatwaves
        temp_boost = 6.0 if active_weather == "Heatwave" else 0.0
        local_temp = np.clip(np.random.normal(annual_temp_cycle + temp_boost, 1.8), 16, 48)

        #patient admissions calculation
        patient_surge = 0

        #winter increases seasonal flu admissions
        if is_winter:
                patient_surge += 25

        #severe weather events cause cumulative strain
        if weather_alert == 1:
            patient_surge +=45

        #calcualte total admitted patients for the day using nd
        total_admitted_patients = int(max(30, np.random.normal(base_patients + patient_surge, 12)))

        #dictionary for the day's record
        daily_record = {
            "Timestamp": date.strftime("%Y-%m-%d"),
            "Facility_ID": facility,
            "Local_Temp_C": round(local_temp, 1),
            "Seasonal_Flu_Rate": round(flu_rate, 2),
            "Weather_Alert_Flag": weather_alert,
            "Weather_Type": weather_type,
            "Admitted_Patients": total_admitted_patients,
        }

        for item in items:
            #intventory consumption logic
            if item == "N95_Masks":
                consumption_per_patient = 1.2
                item_base_spike = np.random.randint(20, 40) if weather_type == "Sandstorm/Dust_Storm" else 0
            elif item == "IV_Fluids":
                consumption_per_patient = 2.5
                item_base_spike = (np.random.randint(50,80) if weather_type == "Heatwave" else 0)
                #spike during heatwave
            elif item == "Inhalers":
                consumption_per_patient = 0.9
                item_base_spike = (np.random.randint(30,50) if weather_type == "Sandstorm/Dust_Storm" else 0)
                #spike during dust storms
            elif item == "Insulin_Vials":
                consumption_per_patient = 0.4
                item_base_spike = 0 #steady chronic demand
            elif item == "Antibiotics":
                consumption_per_patient = 1.8
                item_base_spike = 15 if weather_type == "Heatwave" else 10
            else: #painkillers
                consumption_per_patient = 0.8
                item_base_spike = 15 if is_winter else 0

            #final units used 
            noise = np.random.normal(0,9)
            units_used = int((total_admitted_patients * consumption_per_patient) + item_base_spike + noise)
            units_used = max(10, units_used) #inventory use should never be negative

            #assign to column name
            daily_record[f'Units_{item}'] = units_used

        #append to list
        data.append(daily_record)

#convert master list into pandas dataframe
df_synthetic = pd.DataFrame(data)

#ensure data is sorted before shifting
df_synthetic["Timestamp"] = pd.to_datetime(df_synthetic["Timestamp"])
df_synthetic = df_synthetic.sort_values(by=["Facility_ID", "Timestamp"]).reset_index(drop=True)

#define number of days in the future shouldbe forecasted
forecast_horizon = 7

#shift target item columns backawards within groups to represent future demand 
for item in items:
    df_synthetic[f"Target_{item}_7d_Ahead"] = (
        df_synthetic.groupby("Facility_ID")[f"Units_{item}"]
        .shift(-forecast_horizon)
    )

#last few rows dropped
df_synthetic = df_synthetic.dropna().reset_index(drop=True)

#export dataframe into a csv file saved in project folder
df_synthetic.to_csv("StockPulse_synthetic_data.csv", index = False)

print("Success " + str(len(df_synthetic)) + " rows of expanded synthetic healthcare data")


