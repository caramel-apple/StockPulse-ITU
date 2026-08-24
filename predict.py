#imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

#load dataset
df = pd.read_csv("StockPulse_synthetic_data.csv")

#sorting data
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='mixed', dayfirst=True)
df = df.sort_values(by=["Facility_ID", "Timestamp"]).reset_index(drop=True)
   

#define items
items = ["N95_Masks", "IV_Fluids", "Antibiotics", "Inhalers", "Insulin_Vials", "Painkillers"]

#create lag and rolling features
for item in items:
    consumption_col = f'Units_{item}'

    #lag feature (consumption 7 days ago)
    df[f"{item}_Lag_7d"] = df.groupby("Facility_ID")[consumption_col].shift(7)

    #rolling feature (7-day avg consumption)
    df[f"{item}_Rolling_7d"] = df.groupby("Facility_ID")[consumption_col].rolling(window=7).mean().reset_index(0, drop=True)

#drop rows with NaN values due to shifting process
df = df.dropna().reset_index(drop=True)

#define base features
base_features = ["Local_Temp_C", "Seasonal_Flu_Rate", "Weather_Alert_Flag", "Admitted_Patients"]

models = {}

print("--- Training models---")

for item in items:
    target = f"Target_{item}_7d_Ahead"

    item_features =base_features + [f"{item}_Lag_7d", f"{item}_Rolling_7d"]
    
    X = df[item_features]
    y = df[target]
    
    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #train the random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #feature importances
    importances = model.feature_importances_
    feature_names = item_features if 'item_features' in locals() else X.columns

    feature_importance_df = pd.DataFrame({ 'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print(f"\n--- Top Drivers for {item} ---")
    print(feature_importance_df.head(3))

    
    #evaluate performance
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model for {item} trained successfully! (MAE: {mae:.2f} units)")
    
    #save model
    models[item] = model
    joblib.dump(model, f"model_{item}.pkl")

print("\nAll models trained and saved successfully as .pkl files!")