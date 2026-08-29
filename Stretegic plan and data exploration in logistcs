# First-week-
Logistics Data Analyst Intern


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on Earth."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


# 1. Feature Engineering & Preprocessing
df = pd.read_csv("logistics_delivery_data.csv")
df["distance_km"] = haversine_np(
    df["hub_lon"], df["hub_lat"], df["drop_lon"], df["drop_lat"]
)
df["dispatch_hour"] = pd.to_datetime(df["dispatch_time"]).dt.hour

# 2. Zone Clustering for Fleet Assignment
coordinates = df[["drop_lat", "drop_lon"]]
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
df["delivery_zone"] = kmeans.fit_predict(coordinates)

# 3. Delivery Duration Prediction Model
features = ["distance_km", "dispatch_hour", "package_weight_kg", "traffic_index"]
X = df[features]
y = df["actual_delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Baseline R^2: {model.score(X_test, y_test):.4f}")
