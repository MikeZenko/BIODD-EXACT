#Machine learning model using the ChEMBL bioactivity data for pancreatic cancer drug discovery

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, RFECV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
import sys
import pickle
from scipy.stats import pearsonr
import joblib

print("Step 4: Regression Models with Random Forest")

#Load the dataset prepared in Step 3
input_file = 'pancreatic_cancer_bioactivity_data_3class_pIC50_pubchem_fp.csv'
if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} not found! Run Step 3 first.")
    sys.exit(1)

try:
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with {df.shape[0]} compounds and {df.shape[1]} features")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    sys.exit(1)

#Check if 'Name' column exists and remove it if present
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])
    print("Removed 'Name' column")

#Input features - all columns except pIC50
if 'pIC50' not in df.columns:
    print("ERROR: Dataset doesn't contain 'pIC50' column")
    sys.exit(1)
    
X = df.drop('pIC50', axis=1)
print(f"Input features: {X.shape[1]} descriptors")

#Output variable - pIC50 values
Y = df.pIC50
print(f"Output variable: pIC50 with range {Y.min():.2f} to {Y.max():.2f}")

# Separate Morgan fingerprints from physicochemical descriptors
morgan_cols = [col for col in X.columns if col.startswith('Morgan_')]
physchem_cols = [col for col in X.columns if not col.startswith('Morgan_')]
print(f"Morgan fingerprint bits: {len(morgan_cols)}")
print(f"Physicochemical descriptors: {len(physchem_cols)}")

#Examine the data dimension
print(f"Data dimensions - X: {X.shape}, Y: {Y.shape}")

#Create directory for model outputs
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#Data split (80/20 ratio) with stratification based on pIC50 quantiles
# Create quantile bins for stratification
n_bins = min(5, len(np.unique(Y)))
Y_bins = pd.qcut(Y, q=n_bins, labels=False, duplicates='drop')
X_train, X_test, Y_train, Y_test, Y_bins_train, Y_bins_test = train_test_split(
    X, Y, Y_bins, test_size=0.2, random_state=42, stratify=Y_bins
)
# Convert to numpy arrays for easier indexing
Y_bins_train = np.array(Y_bins_train)
print(f"Training set: {X_train.shape[0]} compounds, Test set: {X_test.shape[0]} compounds")

# Feature correlation analysis for physicochemical descriptors
if len(physchem_cols) > 0:
    print("Analyzing feature correlations...")
    X_physchem = X_train[physchem_cols]
    corr_matrix = X_physchem.corr().abs()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='viridis', square=True, annot=False, 
                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.tight_layout()
    plt.savefig('plots/physicochemical_correlation.png', dpi=300)
    plt.savefig('plots/physicochemical_correlation.pdf')
    plt.close()
    
    # Calculate correlation with target
    target_corr = {}
    for col in physchem_cols:
        corr, _ = pearsonr(X_train[col], Y_train)
        target_corr[col] = abs(corr)
    
    # Plot top correlations with target
    target_corr = pd.Series(target_corr).sort_values(ascending=False)
    top_corr = target_corr.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top 15 Physicochemical Descriptors Correlated with pIC50')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig('plots/top_correlations.png', dpi=300)
    plt.savefig('plots/top_correlations.pdf')
    plt.close()
    
    print(f"Generated correlation plots in plots/ directory")

# Create preprocessing pipeline with feature selection
print("Creating preprocessing and feature selection pipeline...")

# Separate preprocessing for fingerprints and physicochemical descriptors
# For fingerprints: PCA to reduce dimensionality
# For physicochemical: Scale and select based on mutual information
morgan_transformer = Pipeline([
    ('pca', PCA(n_components=0.95))  # Keep components explaining 95% of variance
])

# Modified to use n_jobs=1 to avoid parallel processing
physchem_transformer = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', RFECV(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1), 
        step=2, 
        cv=5,
        scoring='neg_mean_squared_error',
        min_features_to_select=5
    ))
])

# Transform the data
if len(morgan_cols) > 0:
    print("Transforming Morgan fingerprints with PCA...")
    X_train_morgan = morgan_transformer.fit_transform(X_train[morgan_cols])
    print(f"Reduced Morgan fingerprints from {len(morgan_cols)} to {X_train_morgan.shape[1]} components")
    X_test_morgan = morgan_transformer.transform(X_test[morgan_cols])
else:
    X_train_morgan = np.empty((X_train.shape[0], 0))
    X_test_morgan = np.empty((X_test.shape[0], 0))

if len(physchem_cols) > 0:
    print("Selecting important physicochemical descriptors...")
    # First apply scaling
    scaler = StandardScaler()
    scaled_physchem = scaler.fit_transform(X_train[physchem_cols])
    
    # Then apply mutual information instead of RFECV which is too slow
    print("Using mutual information for feature selection (faster than RFECV)...")
    mi_scores = mutual_info_regression(scaled_physchem, Y_train)
    mi_threshold = np.percentile(mi_scores, 70)  # Keep top 30%
    selected_mask = mi_scores >= mi_threshold
    selected_physchem = [col for selected, col in zip(selected_mask, physchem_cols) if selected]
    print(f"Selected {len(selected_physchem)} physicochemical descriptors using mutual information")
    
    # Transform data based on selection
    X_train_physchem = scaled_physchem[:, selected_mask]
    X_test_physchem = scaler.transform(X_test[physchem_cols])[:, selected_mask]
else:
    X_train_physchem = np.empty((X_train.shape[0], 0))
    X_test_physchem = np.empty((X_test.shape[0], 0))

# Combine transformed features
X_train_processed = np.hstack([X_train_morgan, X_train_physchem])
X_test_processed = np.hstack([X_test_morgan, X_test_physchem])
print(f"Final processed features: {X_train_processed.shape[1]}")

# Save the transformers for future use
joblib.dump(morgan_transformer, 'models/morgan_transformer.pkl')
joblib.dump(selected_mask, 'models/physchem_selector.pkl')
joblib.dump(scaler, 'models/physchem_scaler.pkl')

# Nested Cross-Validation to evaluate model performance more reliably
print("Performing nested cross-validation with proper data isolation...")

# Parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],  # Reduced from [100, 200, 300]
    'max_depth': [None, 10, 20],  # Removed 30
    'min_samples_split': [2, 5],  # Removed 10
    'min_samples_leaf': [1, 2],   # Removed 4
    'bootstrap': [True]           # Only use True to reduce parameter combinations
}

# Outer and inner CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Lists to store results
nested_scores = []

# Create a completely separate test set for final evaluation (20% of data)
X_dev, X_holdout, Y_dev, Y_holdout, Y_bins_dev, Y_bins_holdout = train_test_split(
    X, Y, Y_bins, test_size=0.2, random_state=42, stratify=Y_bins
)
# Convert to numpy arrays for easier indexing
Y_bins_dev = np.array(Y_bins_dev)

# Perform nested CV on the development set only
for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_dev, Y_bins_dev)):
    print(f"Fold {i+1}/5:")
    # Split data
    X_train_outer, X_val_outer = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
    y_train_outer, y_val_outer = Y_dev.iloc[train_idx], Y_dev.iloc[val_idx]
    
    # IMPORTANT: Apply feature processing inside the CV loop to prevent data leakage
    print("  Performing feature processing within fold...")
    
    # Process Morgan fingerprints - recalculate PCA within fold
    morgan_fold = Pipeline([
        ('pca', PCA(n_components=0.95))
    ])
    X_train_morgan = morgan_fold.fit_transform(X_train_outer[morgan_cols])
    X_val_morgan = morgan_fold.transform(X_val_outer[morgan_cols])
    
    # Process physicochemical descriptors - rescale within fold
    scaler_fold = StandardScaler()
    X_train_physchem_scaled = scaler_fold.fit_transform(X_train_outer[physchem_cols])
    X_val_physchem_scaled = scaler_fold.transform(X_val_outer[physchem_cols])
    
    # Select features independently within each fold
    print("  Selecting features within fold...")
    
    # Feature selection within fold using mutual information
    mi_scores = mutual_info_regression(X_train_physchem_scaled, y_train_outer)
    mi_threshold = np.percentile(mi_scores, 70)  # Keep top 30%
    selected_features_mask = mi_scores >= mi_threshold
    
    # Apply feature selection
    X_train_physchem = X_train_physchem_scaled[:, selected_features_mask]
    X_val_physchem = X_val_physchem_scaled[:, selected_features_mask]
    
    # Combine features
    X_train_processed = np.hstack([X_train_morgan, X_train_physchem])
    X_val_processed = np.hstack([X_val_morgan, X_val_physchem])
    
    print(f"  Features after selection: {X_train_processed.shape[1]}")
    
    # Define grid search - set n_jobs=1 to avoid parallel processing
    model = RandomForestRegressor(random_state=42, n_jobs=1)
    
    print("  Running grid search with simplified parameters...")
    # Use a simplified grid search to speed up the process
    simple_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=simple_grid,
        cv=inner_cv.split(X_train_processed, Y_bins_dev[train_idx]),
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    # Fit on outer training data
    grid_search.fit(X_train_processed, y_train_outer)
    
    # Predict on validation data
    y_pred = grid_search.predict(X_val_processed)
    
    # Evaluate
    score = r2_score(y_val_outer, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_outer, y_pred))
    nested_scores.append(score)
    print(f"  Fold R² = {score:.4f}, RMSE = {rmse:.4f}")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Progress: {i+1}/5 folds completed ({(i+1)/5*100:.1f}%)")

print(f"Nested CV R² scores: {nested_scores}")
print(f"Mean nested CV R²: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")

# Now prepare the final model on all development data
print("\nTraining final model on all development data for holdout evaluation...")

# Process features for development and holdout sets
# PCA for Morgan fingerprints
morgan_final = Pipeline([
    ('pca', PCA(n_components=0.95))
])
X_dev_morgan = morgan_final.fit_transform(X_dev[morgan_cols])
X_holdout_morgan = morgan_final.transform(X_holdout[morgan_cols])

# Scale physicochemical descriptors
scaler_final = StandardScaler()
X_dev_physchem_scaled = scaler_final.fit_transform(X_dev[physchem_cols])
X_holdout_physchem_scaled = scaler_final.transform(X_holdout[physchem_cols])

# Feature selection on development set
mi_scores_final = mutual_info_regression(X_dev_physchem_scaled, Y_dev)
mi_threshold_final = np.percentile(mi_scores_final, 70)  # Keep top 30%
selected_features_mask_final = mi_scores_final >= mi_threshold_final

# Apply feature selection
X_dev_physchem = X_dev_physchem_scaled[:, selected_features_mask_final]
X_holdout_physchem = X_holdout_physchem_scaled[:, selected_features_mask_final]

# Combine features
X_dev_processed = np.hstack([X_dev_morgan, X_dev_physchem])
X_holdout_processed = np.hstack([X_holdout_morgan, X_holdout_physchem])

print(f"Final feature count: {X_dev_processed.shape[1]}")

# Hyperparameter tuning on development set
print("Performing hyperparameter tuning on development set...")
print("Using simplified parameters to avoid memory issues...")

# Simplified parameters for final model
simplified_param_grid = {
    'n_estimators': [100],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}

grid_search_final = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=1),
    param_grid=simplified_param_grid,
    cv=3,  # Reduced from 5
    scoring='r2',
    n_jobs=1,
    verbose=1
)

# Train model with hyperparameter tuning
print("Fitting final Random Forest model...")
grid_search_final.fit(X_dev_processed, Y_dev)

# Get best parameters and model
best_params = grid_search_final.best_params_
print(f"Best parameters: {best_params}")
rf_model = grid_search_final.best_estimator_

# Train individual models for ensemble
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

print("Training ensemble of models...")
# Define models for ensemble with regularization to prevent overfitting
ridge = Ridge(alpha=10.0, random_state=42)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=3000)
svr = SVR(C=1.0, epsilon=0.2, kernel='rbf')
knn = KNeighborsRegressor(n_neighbors=7, weights='distance')

# Train individual models one by one with progress tracking
print("Training Ridge model...")
ridge.fit(X_dev_processed, Y_dev)
print("Training ElasticNet model...")
elastic.fit(X_dev_processed, Y_dev)
print("Training SVR model...")
svr.fit(X_dev_processed, Y_dev)
print("Training KNN model...")
knn.fit(X_dev_processed, Y_dev)

# Create voting ensemble with n_jobs=1
ensemble = VotingRegressor([
    ('rf', rf_model),
    ('ridge', ridge),
    ('elastic', elastic),
    ('svr', svr),
    ('knn', knn)
], n_jobs=1)  # Set n_jobs=1 to avoid parallel processing

# Train the ensemble
print("Training ensemble model...")
ensemble.fit(X_dev_processed, Y_dev)

#Evaluate both models on development and holdout sets
# Random Forest predictions
print("Generating predictions...")
rf_y_pred_dev = rf_model.predict(X_dev_processed)
rf_y_pred_holdout = rf_model.predict(X_holdout_processed)

# Ensemble predictions
ens_y_pred_dev = ensemble.predict(X_dev_processed)
ens_y_pred_holdout = ensemble.predict(X_holdout_processed)

#Calculate metrics for both models
# Random Forest metrics
rf_dev_r2 = r2_score(Y_dev, rf_y_pred_dev)
rf_holdout_r2 = r2_score(Y_holdout, rf_y_pred_holdout)
rf_dev_rmse = np.sqrt(mean_squared_error(Y_dev, rf_y_pred_dev))
rf_holdout_rmse = np.sqrt(mean_squared_error(Y_holdout, rf_y_pred_holdout))
rf_dev_mae = mean_absolute_error(Y_dev, rf_y_pred_dev)
rf_holdout_mae = mean_absolute_error(Y_holdout, rf_y_pred_holdout)

# Ensemble metrics
ens_dev_r2 = r2_score(Y_dev, ens_y_pred_dev)
ens_holdout_r2 = r2_score(Y_holdout, ens_y_pred_holdout)
ens_dev_rmse = np.sqrt(mean_squared_error(Y_dev, ens_y_pred_dev))
ens_holdout_rmse = np.sqrt(mean_squared_error(Y_holdout, ens_y_pred_holdout))
ens_dev_mae = mean_absolute_error(Y_dev, ens_y_pred_dev)
ens_holdout_mae = mean_absolute_error(Y_holdout, ens_y_pred_holdout)

print(f"\nModel performance with proper data isolation:")
print(f"Random Forest - Development: R²={rf_dev_r2:.4f}, RMSE={rf_dev_rmse:.4f}, MAE={rf_dev_mae:.4f}")
print(f"Random Forest - Holdout: R²={rf_holdout_r2:.4f}, RMSE={rf_holdout_rmse:.4f}, MAE={rf_holdout_mae:.4f}")
print(f"Ensemble - Development: R²={ens_dev_r2:.4f}, RMSE={ens_dev_rmse:.4f}, MAE={ens_dev_mae:.4f}")
print(f"Ensemble - Holdout: R²={ens_holdout_r2:.4f}, RMSE={ens_holdout_rmse:.4f}, MAE={ens_holdout_mae:.4f}")
print(f"Nested CV R²: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")

# Compare nested CV with holdout to verify consistency
print(f"\nValidation consistency check:")
print(f"Nested CV mean R²: {np.mean(nested_scores):.4f}")
print(f"Random Forest holdout R²: {rf_holdout_r2:.4f}")
print(f"Ensemble holdout R²: {ens_holdout_r2:.4f}")
if abs(np.mean(nested_scores) - rf_holdout_r2) > 0.2:
    print("WARNING: Large gap between nested CV and holdout performance suggests potential issues")
else:
    print("Validation consistency check passed - nested CV score is similar to holdout performance")

#Save model performance metrics
metrics = pd.DataFrame({
    'Metric': ['R-squared', 'RMSE', 'MAE', 'CV R-squared (mean)', 'CV R-squared (std)'],
    'RF_Development': [rf_dev_r2, rf_dev_rmse, rf_dev_mae, np.mean(nested_scores), np.std(nested_scores)],
    'RF_Holdout': [rf_holdout_r2, rf_holdout_rmse, rf_holdout_mae, np.nan, np.nan],
    'Ensemble_Development': [ens_dev_r2, ens_dev_rmse, ens_dev_mae, np.nan, np.nan],
    'Ensemble_Holdout': [ens_holdout_r2, ens_holdout_rmse, ens_holdout_mae, np.nan, np.nan]
})
metrics.to_csv('models/model_metrics.csv', index=False)

#Save the models and transformers for future use
joblib.dump(morgan_final, 'models/morgan_transformer.pkl')
joblib.dump(scaler_final, 'models/physchem_scaler.pkl')
joblib.dump(selected_features_mask_final, 'models/selected_features_mask.pkl')
pickle.dump(rf_model, open('models/random_forest_model.pkl', 'wb'))
pickle.dump(ensemble, open('models/ensemble_model.pkl', 'wb'))
print("Models saved to models/ directory")

#Scatter Plots of Experimental vs Predicted pIC50 Values
plt.figure(figsize=(15, 10))

# Random Forest
plt.subplot(2, 2, 1)
sns.regplot(x=Y_dev, y=rf_y_pred_dev, scatter_kws={'alpha':0.4})
plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
plt.title('Random Forest - Development Set')
plt.plot([Y_dev.min(), Y_dev.max()], [Y_dev.min(), Y_dev.max()], 'k--', lw=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.annotate(f'R² = {rf_dev_r2:.4f}\nRMSE = {rf_dev_rmse:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction', 
            ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

plt.subplot(2, 2, 2)
sns.regplot(x=Y_holdout, y=rf_y_pred_holdout, scatter_kws={'alpha':0.4})
plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
plt.title('Random Forest - Holdout Set')
plt.plot([Y_holdout.min(), Y_holdout.max()], [Y_holdout.min(), Y_holdout.max()], 'k--', lw=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.annotate(f'R² = {rf_holdout_r2:.4f}\nRMSE = {rf_holdout_rmse:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction', 
            ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

# Ensemble
plt.subplot(2, 2, 3)
sns.regplot(x=Y_dev, y=ens_y_pred_dev, scatter_kws={'alpha':0.4})
plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
plt.title('Ensemble - Development Set')
plt.plot([Y_dev.min(), Y_dev.max()], [Y_dev.min(), Y_dev.max()], 'k--', lw=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.annotate(f'R² = {ens_dev_r2:.4f}\nRMSE = {ens_dev_rmse:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction', 
            ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

plt.subplot(2, 2, 4)
sns.regplot(x=Y_holdout, y=ens_y_pred_holdout, scatter_kws={'alpha':0.4})
plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
plt.title('Ensemble - Holdout Set')
plt.plot([Y_holdout.min(), Y_holdout.max()], [Y_holdout.min(), Y_holdout.max()], 'k--', lw=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.annotate(f'R² = {ens_holdout_r2:.4f}\nRMSE = {ens_holdout_rmse:.4f}', 
            xy=(0.05, 0.95), xycoords='axes fraction', 
            ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

plt.tight_layout()
plt.savefig('plots/regression_plots.png', dpi=300)
plt.savefig('plots/regression_plots.pdf')
plt.close()

print("Regression plots saved to plots/regression_plots.png and plots/regression_plots.pdf")

#Feature importance analysis
if hasattr(rf_model, 'feature_importances_'):
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create feature names
    feature_names = []
    feature_idx = 0
    
    # PCA components for Morgan fingerprints
    morgan_components = X_dev_morgan.shape[1]
    for i in range(morgan_components):
        feature_names.append(f"PC{i+1}_Morgan")
        feature_idx += 1
        
    # Selected physicochemical descriptors
    selected_physchem = [col for selected, col in zip(selected_features_mask_final, physchem_cols) if selected]
    for col in selected_physchem:
        feature_names.append(col)
        feature_idx += 1
    
    # Convert the feature importances to DataFrame
    feature_importance = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances}
    )
    
    # Sort the dataframe by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.savefig('plots/feature_importance.pdf')
    plt.close()
    
    print("Feature importance plot saved to plots/feature_importance.png")
    
    # Save feature importances
    feature_importance.to_csv('models/feature_importance.csv', index=False)

print("\nStep 4 completed. Run 5.Comparing_Regressors.py next.")


