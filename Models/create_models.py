import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load the data
print("Loading data...")
df = pd.read_excel("Knowledge_base/market_data_with_features.xlsx", index_col=0)

# Map horizons to target columns and their specific parameters
horizon_map = {
    '1M': {'target': 'Target_1M', 'alpha': 0.01},
    '3M': {'target': 'Target_3M', 'alpha': 0.1},   # Further increased alpha
    '1Y': {'target': 'Target_1Y', 'alpha': 0.1},
    '2Y': {'target': 'Target_2Y', 'alpha': 0.1},
    '4Y': {'target': 'Target_4Y', 'alpha': 0.1}
}

# Drop target columns and regime column from features
feature_cols = df.drop(['Target_1M', 'Target_3M', 'Target_1Y', 'Target_2Y', 'Target_4Y', 'Regime'], axis=1).columns

# Fill NaN values in features with 0
print("Preprocessing features...")
X = df[feature_cols].fillna(0)

# Create and save models for each horizon
for horizon, params in horizon_map.items():
    print(f"\nTraining model for {horizon}...")
    
    # Create target and remove NaN values
    y = df[params['target']]
    mask = ~y.isna()
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Training data shape for {horizon}: {X_clean.shape}")
    print(f"Number of samples for {horizon}: {len(y_clean)}")
    
    # Create pipeline with scaling and Lasso
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(
            alpha=params['alpha'],
            max_iter=20000,  # Further increased iterations
            tol=1e-3,       # Slightly relaxed tolerance
            random_state=42
        ))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_clean, y_clean)
    
    # Print convergence info
    lasso = pipeline.named_steps['lasso']
    print(f"Lasso iterations: {lasso.n_iter_}")
    print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
    
    # Save the model
    joblib.dump(pipeline, f'Models/lasso_macro_forecast_{horizon}.joblib')
    print(f"Saved model for {horizon}")

# Save the scaler and kmeans separately
print("\nTraining scaler...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
joblib.dump(scaler, 'Models/final_fresh_scaler.joblib')
print("Saved scaler")

# Train and save KMeans
print("\nTraining KMeans...")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
joblib.dump(kmeans, 'Models/final_fresh_kmeans.joblib')
print("Saved KMeans")

print("\nAll models, scaler, and kmeans have been saved successfully!") 