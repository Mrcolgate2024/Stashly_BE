import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set random seed for reproducibility
np.random.seed(42)

print("Loading data...")
# Load your dataset here
excel_path = os.path.join(os.path.dirname(__file__), "market_data_with_features.xlsx")
df = pd.read_excel(excel_path)
df = df.sort_values("Date").set_index("Date")
print(f"Initial data shape: {df.shape}")

print("\nGenerating features...")
# Create a copy of the DataFrame to store all features
feature_df = df.copy()
print(f"Feature DataFrame shape after copy: {feature_df.shape}")

# Generate lags more efficiently
lag_months = [1, 3, 6]
lag_features = {}

for col in df.columns:
    for lag in lag_months:
        lag_features[f"{col}_lag{lag}"] = df[col].shift(lag)

print(f"Number of lag features created: {len(lag_features)}")

# Add all lag features at once
feature_df = pd.concat([feature_df, pd.DataFrame(lag_features)], axis=1)
print(f"DataFrame shape after adding lag features: {feature_df.shape}")

# Target
feature_df["MSCI_World_target"] = df["MSCI World"].shift(-1)

# Calculate returns (only for price-based features)
price_cols = [col for col in df.columns if not col.endswith('%')]
returns_df = df[price_cols].pct_change(fill_method=None).add_suffix('_ret_1m')
print(f"Returns DataFrame shape: {returns_df.shape}")

full_df = pd.concat([feature_df, returns_df], axis=1)
print(f"Full DataFrame shape after adding returns: {full_df.shape}")

# Clean data
print("\nCleaning data...")
# Replace inf values with NaN
full_df = full_df.replace([np.inf, -np.inf], np.nan)
print(f"Shape after replacing inf values: {full_df.shape}")

# Define X and y before dropping NaN
X = full_df.drop(columns=["MSCI World", "MSCI_World_target"])
y = full_df["MSCI_World_target"]
print(f"Initial X shape: {X.shape}")
print(f"Initial y shape: {y.shape}")

# Drop market-related columns
market_indices = [
    "NASDAQ", "Portfolio", "EURO STOXX", "OMX30", "Russell", "S&P 500", "MSCI", "PPM",
    "FTSE", "DAX", "CAC", "IBEX", "AEX", "Nikkei", "Hang Seng", "Shanghai", "Bovespa",
    "TSX", "ASX", "KOSPI", "Sensex", "NIFTY", "BSE", "SSE", "SZSE", "TSE", "HSE"
]

# First, identify columns that are direct market indices
market_cols = [col for col in X.columns if any(index in col for index in market_indices)]
print(f"Number of direct market index columns to drop: {len(market_cols)}")
print("Dropping the following market index columns:")
for col in market_cols:
    print(f"- {col}")

# Then, identify and drop any derived features (returns, lags) from market indices
derived_market_cols = []
for col in X.columns:
    # Check if this is a derived feature (return or lag) from a market index
    for index in market_indices:
        if (f"{index}_ret_" in col or  # Returns
            f"{index}_lag" in col or   # Lags
            f"{index}_target" in col): # Targets
            derived_market_cols.append(col)
            break

print(f"\nNumber of derived market features to drop: {len(derived_market_cols)}")
print("Dropping the following derived market features:")
for col in derived_market_cols:
    print(f"- {col}")

# Drop all identified market-related columns
all_market_cols = market_cols + derived_market_cols
X = X.drop(columns=all_market_cols)
print(f"\nX shape after dropping market columns: {X.shape}")

# Verify no market data remains
remaining_market_cols = [col for col in X.columns if any(index in col for index in market_indices)]
if remaining_market_cols:
    print("\nWARNING: The following market-related columns were not dropped:")
    for col in remaining_market_cols:
        print(f"- {col}")
else:
    print("\nVerification passed: No market-related columns remain in the feature set")

# Fill NaN values
print("\nFilling NaN values...")
# First, handle quarterly data
for col in X.columns:
    if 'GDP' in col:
        X[col] = X[col].ffill()

# Then fill remaining NaN values with median
X = X.fillna(X.median())

# Verify no NaN values remain
print("\nNaN check after filling:")
nan_count = X.isna().sum().sum()
print(f"Total NaN values remaining: {nan_count}")

print(f"Final feature set shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Sample of feature names: {list(X.columns)[:5]}")

if X.shape[0] == 0:
    raise ValueError("No data left after cleaning. Check the data loading and cleaning steps.")

print("\nTraining KMeans for regime detection...")
# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for clustering
n_components = min(50, X.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio with {n_components} components: {pca.explained_variance_ratio_.sum():.3f}")

# Fit KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
regimes = kmeans.fit_predict(X_pca)
full_df.loc[X.index, 'Regime'] = regimes

# Save scaler, PCA, and kmeans
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'final_fresh_scaler.joblib'))
joblib.dump(pca, os.path.join(os.path.dirname(__file__), 'final_fresh_pca.joblib'))
joblib.dump(kmeans, os.path.join(os.path.dirname(__file__), 'final_fresh_kmeans.joblib'))

# Plot regimes
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['MSCI World'], label='MSCI World', alpha=0.5)
for i in range(n_clusters):
    mask = full_df['Regime'] == i
    plt.scatter(full_df.index[mask], df.loc[full_df.index[mask], 'MSCI World'], 
               label=f'Regime {i}', alpha=0.7)
plt.title('Market Regimes Over Time')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'regime_plot.png'))
plt.close()

print("\nTraining models for different horizons...")
# Define horizons
horizons = ['1M', '3M', '1Y', '2Y', '4Y']

# Create one-hot encoding for regimes
regime_dummies = pd.get_dummies(full_df.loc[X.index, 'Regime'], prefix='Regime')
X_with_regime = pd.concat([X, regime_dummies], axis=1)

# Train models for each horizon
for horizon in horizons:
    print(f"\nTraining model for {horizon} horizon...")
    # Create target with appropriate shift
    shift_map = {'1M': 1, '3M': 3, '1Y': 12, '2Y': 24, '4Y': 48}
    y_horizon = df['MSCI World'].shift(-shift_map[horizon])
    
    # Remove rows with NaN targets
    mask = ~y_horizon.loc[X_with_regime.index].isna()
    X_train = X_with_regime[mask]
    y_train = y_horizon.loc[X_with_regime.index][mask]
    
    print(f"Training data shape for {horizon}: {X_train.shape}")
    
    # Create pipeline with PCA and Lasso
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(50, X_train.shape[0] - 1))),
        ('lasso', Lasso(alpha=0.1, max_iter=10000, tol=1e-2))
    ])
    
    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='r2')
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Fit final model
    pipeline.fit(X_train, y_train)
    
    # Save model
    joblib.dump(pipeline, os.path.join(os.path.dirname(__file__), f'lasso_macro_forecast_{horizon}.joblib'))
    
    # Calculate and print metrics
    y_pred = pipeline.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    
    print(f"Training RMSE: {rmse:.2f}")
    print(f"Training R2 Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Actual')
    plt.plot(y_train.index, y_pred, label='Predicted')
    plt.title(f'MSCI World Forecast - {horizon} Horizon')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'forecast_{horizon}.png'))
    plt.close()

print("\nTraining complete! Models and plots have been saved.") 