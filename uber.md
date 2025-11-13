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
# Task 1: Load, Explore, and Pre-process Dataset
# -----------------------------------------------------------------
try:
    df = pd.read_csv('uber.csv')
except FileNotFoundError:
    print("Error: 'uber.csv' not found. Please upload the file if using Colab.")
    raise

# Drop columns that are just identifiers
df.drop(columns=['Unnamed: 0', 'key'], inplace=True, errors='ignore')

# --- Exploratory Data Analysis (EDA) ---
# Run THIS BLOCK FIRST to see the raw data, including NaNs
print("\n--- Initial Data Exploration (Before Cleaning) ---")

# 1. Look at the first few rows
print("--- df.head() ---")
print(df.head())

# 2. Check data types and non-null counts (This will show missing values)
print("\n--- df.info() ---")
df.info()

# 3. Get statistical summary (count will be lower for columns with NaNs)
print("\n--- df.describe() ---")
print(df.describe())
# From .describe(), we can see:
# - fare_amount has a min <= 0 (bad) and a very high max
# - passenger_count has a max of 208 (impossible)
# - Coordinates (lat/lon) have strange values (e.g., min 0)

# --- Pre-processing: Cleaning ---

# 1. Handle missing values (based on what we saw in df.info())
print(f"\nMissing values before dropping: {df.isnull().sum().sum()}")
df.dropna(inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")

# 2. Convert 'pickup_datetime' to datetime objects for feature engineering
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# -----------------------------------------------------------------
# Task 2: Identify and remove outliers
# -----------------------------------------------------------------

# --- Task 2a: Visualize Outliers (Before Removal) ---
# We plot these based on what we saw in df.describe()
print("\nVisualizing outliers before removal...")
plt.figure(figsize=(15, 8))

# 1. Visualize Fare Amount
plt.subplot(2, 2, 1)
sns.boxplot(x=df['fare_amount'])
plt.title('Box Plot: Fare Amount (Original)')
plt.xlabel('Fare Amount ($)')

# 2. Visualize Passenger Count
plt.subplot(2, 2, 2)
sns.boxplot(x=df['passenger_count'])
plt.title('Box Plot: Passenger Count (Original)')
plt.xlabel('Passenger Count')

# 3. Visualize Coordinates (NYC Bounding Box)
plt.subplot(2, 2, 3)
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=1, alpha=0.1)
plt.title('Scatter Plot: Pickup Locations (Original)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-75, -73) # Limit view for clarity
plt.ylim(40.4, 41.2)

plt.tight_layout()
plt.show()

# --- Task 2b: Removing Outliers ---
# These rules are set based on the .describe() and boxplots
print("Removing outliers...")

# 1. Fare Amount (must be positive, set reasonable upper limit)
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 300)]

# 2. Passenger Count (standard car limits)
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# 3. Coordinates (filter to a reasonable NYC bounding box)
df = df[(df['pickup_latitude'] >= 40.5) & (df['pickup_latitude'] <= 41.0)]
df = df[(df['pickup_longitude'] >= -74.5) & (df['pickup_longitude'] <= -73.5)]
df = df[(df['dropoff_latitude'] >= 40.5) & (df['dropoff_latitude'] <= 41.0)]
df = df[(df['dropoff_longitude'] >= -74.5) & (df['dropoff_longitude'] <= -73.5)]


# --- Feature Engineering (part of pre-processing) ---

# 1. Calculate distance using the Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of Earth in kilometers
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

# 4. Remove 0-distance trips
print("Visualizing 0-distance trips...")
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['distance_km'])
plt.title('Box Plot: Trip Distance (Before Removing 0-km trips)')
plt.xlabel('Distance (km)')
plt.show()

df = df[df['distance_km'] > 0.01] # Keep trips > 10 meters

# --- Handle Large Dataset: Sub-sample ---
if len(df) > 100000:
    print(f"Dataset large, sampling 100,000 rows.")
    df = df.sample(n=100000, random_state=42)
else:
    print("Dataset size manageable.")

# -----------------------------------------------------------------
# Task 3: Check the correlation
# -----------------------------------------------------------------
print("Generating correlation heatmap...")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# -----------------------------------------------------------------
# Task 4: Implement Linear Regression and Random Forest
# -----------------------------------------------------------------
print("Starting model training...")
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale the data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# --- Model 2: Random Forest Regression ---
rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(X_train_scaled, y_train)
y_pred_rfr = rfr.predict(X_test_scaled)

print("Model training complete.")

# -----------------------------------------------------------------
# Task 5: Evaluate the models and compare scores
# -----------------------------------------------------------------
print("\n--- FINAL MODEL COMPARISON ---")

# --- Linear Regression Performance ---
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# --- Random Forest Performance ---
r2_rfr = r2_score(y_test, y_pred_rfr)
rmse_rfr = np.sqrt(mean_squared_error(y_test, y_pred_rfr))

# --- Comparison ---
# The RMSE value will be in the same unit as the target (dollars)
print(f"Linear Regression   ->   R2 = {r2_lr:.4f}   RMSE = {rmse_lr:.4f} (in $)")
print(f"Random Forest       ->   R2 = {r2_rfr:.4f}   RMSE = {rmse_rfr:.4f} (in $)")

if r2_rfr > r2_lr:
    print("\nRandom Forest performed better.")
else:
    print("\nLinear Regression performed better (or equal).")
