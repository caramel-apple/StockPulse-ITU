#imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
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
    df[f"{item}_Lag_7d"] = df.groupby("Facility_ID")[consumption_col].shift(7)
    df[f"{item}_Rolling_7d"] = (df.groupby("Facility_ID")[consumption_col].transform(lambda x: x.shift(1).rolling(7).mean())
)

#drop rows with NaN values due to shifting process
df = df.dropna().reset_index(drop=True)

#one-hot encoding
df = pd.get_dummies(df, columns=["Facility_ID", "Weather_Type"], dtype=int)

#define base features
base_features = ["Local_Temp_C", "Seasonal_Flu_Rate", "Weather_Alert_Flag", "Admitted_Patients"]
facility_features = [col for col in df.columns if col.startswith("Facility_ID_")]
weather_features = [col for col in df.columns if col.startswith("Weather_Type_")]
base_features = base_features + facility_features + weather_features

#chronological split
split_date = pd.Timestamp("2025-07-01")
train_df = df[df["Timestamp"] < split_date]
test_df = df[df["Timestamp"] >= split_date]

print(f"Training data: {train_df['Timestamp'].min()} to {train_df['Timestamp'].max()}")
print(f"Testing data: {test_df['Timestamp'].min()} to {test_df['Timestamp'].max()}")

models = {}
print("--- Training models---")

for item in items:
    target = f"Target_{item}_7d_Ahead"

    item_features =base_features + [f"{item}_Lag_7d", f"{item}_Rolling_7d"]
    
    X = df[item_features]
    y = df[target]
    
    #train-test split
    X_train = train_df[item_features]
    y_train = train_df[target]
    X_test = test_df[item_features]
    y_test = test_df[target]
    
    #train the random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': item_features, 
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\n--- Top Drivers for {item} ---")
    print(feature_importance_df.head(3).to_string(index=False))

    
    #evaluate performance
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    
    print(f"Model for {item} trained successfully!")
    print(f"MAE:  {mae:.2f} units")
    print(f"MAPE: {mape:.2f}%")
    
    #save model
    models[item] = model
    joblib.dump(model, f"model_{item}.pkl")

print("\nAll models trained and saved successfully as .pkl files!")