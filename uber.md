```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------------------------
# Task 1: Pre-process the dataset
# -----------------------------------------------------------------
print("--- Starting Task 1: Pre-processing Dataset ---")
# Load the dataset
try:
    df = pd.read_csv('uber.csv')
except FileNotFoundError:
    print("Error: 'uber.csv' not found. Please upload the file if using Colab.")
    raise

print(f"Initial dataset shape: {df.shape}")
df.drop(columns=['Unnamed: 0', 'key'], inplace=True, errors='ignore')

# Handle missing values
print(f"Missing values before dropping: {df.isnull().sum().sum()}")
df.dropna(inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")

# Convert 'pickup_datetime' to datetime objects
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# -----------------------------------------------------------------
# Task 2: Identify and remove outliers
# -----------------------------------------------------------------
print("\n--- Starting Task 2: Identifying and Removing Outliers ---")

# NEW: Task 2a - Visualize Outliers (Before Removal)
# We plot these *before* filtering to see what we're removing.
print("Visualizing outliers before removal...")

plt.figure(figsize=(15, 8))

# 1. Visualize Fare Amount
plt.subplot(2, 2, 1)
sns.boxplot(x=df['fare_amount'])
plt.title('Box Plot: Fare Amount (Original)')
plt.xlabel('Fare Amount ($)')
#

# 2. Visualize Passenger Count
plt.subplot(2, 2, 2)
sns.boxplot(x=df['passenger_count'])
plt.title('Box Plot: Passenger Count (Original)')
plt.xlabel('Passenger Count')
#

# 3. Visualize Coordinates (NYC Bounding Box)
# This shows us all the pickup points, including ones outside NYC
plt.subplot(2, 2, 3)
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=1, alpha=0.1)
plt.title('Scatter Plot: Pickup Locations (Original)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-75, -73) # Limit view for clarity
plt.ylim(40.4, 41.2)
#

plt.tight_layout()
plt.show()

# --- Task 2b: Removing Outliers (Your original code) ---
print("Removing outliers based on visualizations...")

# 1. Fare Amount
print(f"Shape before fare_amount filter: {df.shape}")
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 300)]
print(f"Shape after fare_amount filter: {df.shape}")

# 2. Passenger Count
print(f"Shape before passenger_count filter: {df.shape}")
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
print(f"Shape after passenger_count filter: {df.shape}")

# 3. Coordinates
print(f"Shape before coordinate filter: {df.shape}")
df = df[(df['pickup_latitude'] >= 40.5) & (df['pickup_latitude'] <= 41.0)]
df = df[(df['pickup_longitude'] >= -74.5) & (df['pickup_longitude'] <= -73.5)]
df = df[(df['dropoff_latitude'] >= 40.5) & (df['dropoff_latitude'] <= 41.0)]
df = df[(df['dropoff_longitude'] >= -74.5) & (df['dropoff_longitude'] <= -73.5)]
print(f"Shape after coordinate filter: {df.shape}")


# --- Feature Engineering (part of pre-processing) ---
print("\n--- Starting Feature Engineering ---")

# 1. Calculate distance using the Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# 2. Extract features from 'pickup_datetime'
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year

# 3. Drop original columns
df.drop(['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
         'dropoff_latitude', 'dropoff_longitude'], axis=1, inplace=True)

# 4. Remove trips with 0 distance (This is also an outlier check!)

# NEW: Visualize the 0-distance outliers before removing them
print("Visualizing 0-distance trips (outliers)...")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['distance_km'])
plt.title('Box Plot: Trip Distance (Before Removing 0-km trips)')
plt.xlabel('Distance (km)')
plt.show()
#

print(f"Shape before 0-distance filter: {df.shape}")
df = df[df['distance_km'] > 0.01] # Keep trips > 10 meters
print(f"Shape after 0-distance filter: {df.shape}")

# --- Handle Large Dataset: Sub-sample ---
if len(df) > 100000:
    print(f"Dataset is large ({len(df)} rows). Sampling 100,000 rows.")
    df = df.sample(n=100000, random_state=42)
else:
    print("Dataset is of a manageable size.")

print(f"\nFinal pre-processed data head:\n{df.head()}")

# -----------------------------------------------------------------
# Task 3: Check the correlation
# -----------------------------------------------------------------
print("\n--- Starting Task 3: Correlation Check ---")

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()
print("Generated correlation heatmap.")
#

# -----------------------------------------------------------------
# Task 4: Implement Linear Regression and Random Forest
# -----------------------------------------------------------------
print("\n--- Starting Task 4: Model Implementation ---")

# Define features (X) and target (y)
# X will contain all columns EXCEPT 'fare_amount'
# This *includes* 'distance_km', 'hour', 'day_of_week', etc.
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

print(f"Features being used for training: {list(X.columns)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale the data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Linear Regression ---
print("Training Linear Regression model...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Linear Regression training complete.")

# --- Model 2: Random Forest Regression ---
print("Training Random Forest Regression model...")
rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(X_train_scaled, y_train)
y_pred_rfr = rfr.predict(X_test_scaled)
print("Random Forest training complete.")

# -----------------------------------------------------------------
# Task 5: Evaluate the models and compare scores
# -----------------------------------------------------------------
print("\n--- Starting Task 5: Model Evaluation ---")

# --- Linear Regression Evaluation ---
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("\n--- Linear Regression Performance ---")
print(f"R-squared (R2): {r2_lr:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lr:.4f}")

# --- Random Forest Evaluation ---
r2_rfr = r2_score(y_test, y_pred_rfr)
rmse_rfr = np.sqrt(mean_squared_error(y_test, y_pred_rfr))
print("\n--- Random Forest Performance ---")
print(f"R-squared (R2): {r2_rfr:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rfr:.4f}")

# --- Comparison ---
print("\n--- Model Comparison ---")
results = {
    'Model': ['Linear Regression', 'Random Forest'],
    'R-squared (R2)': [r2_lr, r2_rfr],
    'RMSE': [rmse_lr, rmse_rfr]
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

if r2_rfr > r2_lr:
    print("\nThe Random Forest model performed better (higher R2 score and lower RMSE).")
else:
    print("\nThe Linear Regression model performed better (higher R2 score and lower RMSE).")
    ```
