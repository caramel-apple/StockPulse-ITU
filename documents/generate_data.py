#in terminal write "python -m pip install pandas numpy"

#import libraries
import numpy as np
import pandas as pd

#data is reproducible
np.random.seed(42)

#define the timeframe between jan 2024 - jan 2026
dates = pd.date_range(start="2024-01-01", end = "2026-01-01", freq = "D")

#define 3 simulated healthcare facilities 
facilities = ["HOSP_RUH_01", "HOSP_JED_02", "HOSP_DMM_03"]

#define medical supply items 
items = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

#initialize empty list to hold data
data = []

#loop through every facility, item type, day
for facility in facilities:
    for item in items:
        for date in dates:
            #extract motnth, year to calculate season
            month = date.month

            #determine if current month is in winter (Nov-Feb)
            is_winter = month in [11, 12, 1, 2]

            #simulate regional flu rates (higher in winter)
            flu_rate = np.random.normal(50,10) + (25 if is_winter else 0)

            #simulate local temp (in celsius) higher in summer (Jun-Aug)
            base_temp = 38 if month in [6, 7, 8] else 23
            local_temp = np.random.normal(base_temp, 4)

            #simulate severe weather events (5% chance of sandstorm daily)
            weather_alert = 1 if np.random.rand() < 0.05 else 0

            #determine weather event ype based on temp and alert flag
            if weather_alert == 1:
                weather_type = (
                    "Heatwave" if local_temp > 40 else "Sandstorm/Dust_Storm"

                )
            else:
                weather_type = "None"


            #patient admissions
            #base patient census for a hospital
            base_patients = 120
            patient_surge = 0

            #winter increases seasonal flu admissions
            if is_winter:
                patient_surge += 25

            #severe weather events cause emergency department surge (heatstroke, asthma)
            if weather_alert == 1:
                patient_surge +=40

            #calcualte total admitted patients for the day using nd
            total_admitted_patients = int(max(50, np.random.normal(base_patients + patient_surge, 12)))

            #intventory consumption logic
            if item == "N95_Masks":
                consumption_per_patient = 1.2
                item_base_spike = 30 if weather_type == "Sandstorm/Dust_Storm" else 0
            elif item == "IV_Fluids":
                consumption_per_patient = 2.5
                item_base_spike = 60 if weather_type == "Heatwave" else -0
            elif item == "Inhalers":
                comsuption_per_patient = 0.9
                item_base_spike = (40 if weather_type == "Sandstorm/Dust_Storm" else 0)
                #spike during dust storms
            elif item == "Insulin_Vials":
                consumption_per_patient = 0.4
                item_base_spike = 0 #steady chronic demand
            elif item == "Antibiotics":
                consumption_per_patient = 1.8
                item_base_spike = 10 #genera; high daily usage
            else: #antibiotics
                consumption_per_patient = 0.8
                item_base_spike = 15 if is_winter else 0

            #final units used 
            noise = np.random.normal(0.10)
            units_used = int((total_admitted_patients * consumption_per_patient) + item_base_spike + noise)

            units_used = max(10, units_used) #inventory use should never be negative

            #append structured record to data list
            data.append({
                "Timestamp": date.strftime("%Y-%m-%d"),
                "Facility_ID": facility,
                "Supply_Item_Type": item,
                "Local_Temp_C": round(local_temp, 1),
                "Seasonal_Flu_Rate": round(flu_rate, 2),
                "Weather_Alert_Flag": weather_alert,
                "Weather_Type": weather_type,
                "Admitted_Patients": total_admitted_patients,
                "Units_Used": units_used,
            })

#convert master list into pandas dataframe
df_synthetic = pd.DataFrame(data)

#export dataframe into a csv file saved in project folder
df_synthetic.to_csv("StockPulse_synthetic_data.csv", index = False)

print("Success " + str(len(df_synthetic)) + " rows of expanded synthetic healthcare data")


