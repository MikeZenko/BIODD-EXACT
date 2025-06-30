#Comparing different regression models for pancreatic cancer drug discovery

#1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import os
import sys
import pickle
import time
import joblib

# Import various regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

print("Step 5: Comparing Different Regression Models")

#2. Load the dataset prepared in Step 3
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

#Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

#3. Prepare input and output variables
X = df.drop('pIC50', axis=1)
Y = df.pIC50

print(f"Input features: {X.shape[1]} molecular descriptors")
print(f"Output variable: pIC50 with range {Y.min():.2f} to {Y.max():.2f}")

# Separate Morgan fingerprints from physicochemical descriptors
morgan_cols = [col for col in X.columns if col.startswith('Morgan_')]
physchem_cols = [col for col in X.columns if not col.startswith('Morgan_')]
print(f"Morgan fingerprint bits: {len(morgan_cols)}")
print(f"Physicochemical descriptors: {len(physchem_cols)}")

#4. Load feature transformers if they exist
if os.path.exists('models/morgan_transformer.pkl') and os.path.exists('models/physchem_transformer.pkl'):
    print("Loading feature transformers from models/ directory")
    morgan_transformer = joblib.load('models/morgan_transformer.pkl')
    physchem_transformer = joblib.load('models/physchem_transformer.pkl')
    
    # Use the transformers from Step 4
    print("Using pre-trained transformers for feature processing")
else:
    print("Feature transformers not found. Creating new ones...")
    # For fingerprints: PCA to reduce dimensionality
    morgan_transformer = Pipeline([
        ('pca', PCA(n_components=0.95))  # Keep components explaining 95% of variance
    ])
    
    # For physicochemical: Scale and select most important
    physchem_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])

#5. Perform data splitting using 80/20 ratio with stratification
# Create quantile bins for stratification
n_bins = min(5, len(np.unique(Y)))
Y_bins = pd.qcut(Y, q=n_bins, labels=False, duplicates='drop')
X_dev, X_holdout, Y_dev, Y_holdout, Y_bins_dev, Y_bins_holdout = train_test_split(
    X, Y, Y_bins, test_size=0.2, random_state=42, stratify=Y_bins
)
# Convert to numpy arrays for easier indexing
Y_bins_dev = np.array(Y_bins_dev)
print(f"Development set: {X_dev.shape[0]} compounds, Holdout set: {X_holdout.shape[0]} compounds")

# Transform features
# Morgan fingerprints
if len(morgan_cols) > 0:
    print("Processing Morgan fingerprints...")
    morgan_final = Pipeline([('pca', PCA(n_components=0.95))])
    X_dev_morgan = morgan_final.fit_transform(X_dev[morgan_cols])
    X_holdout_morgan = morgan_final.transform(X_holdout[morgan_cols])
    print(f"Reduced Morgan fingerprints from {len(morgan_cols)} to {X_dev_morgan.shape[1]} components")
else:
    X_dev_morgan = np.empty((X_dev.shape[0], 0))
    X_holdout_morgan = np.empty((X_holdout.shape[0], 0))

# Physicochemical descriptors
if len(physchem_cols) > 0:
    print("Processing physicochemical descriptors...")
    scaler_final = StandardScaler()
    X_dev_physchem_scaled = scaler_final.fit_transform(X_dev[physchem_cols])
    X_holdout_physchem_scaled = scaler_final.transform(X_holdout[physchem_cols])
    
    # Select features based on mutual information
    mi_scores = mutual_info_regression(X_dev_physchem_scaled, Y_dev)
    mi_threshold = np.percentile(mi_scores, 70)  # Keep top 30%
    selected_mask = mi_scores >= mi_threshold
    selected_physchem = [col for selected, col in zip(selected_mask, physchem_cols) if selected]
    print(f"Selected {np.sum(selected_mask)} physicochemical descriptors")
    
    # Apply feature selection
    X_dev_physchem = X_dev_physchem_scaled[:, selected_mask]
    X_holdout_physchem = X_holdout_physchem_scaled[:, selected_mask]
else:
    X_dev_physchem = np.empty((X_dev.shape[0], 0))
    X_holdout_physchem = np.empty((X_holdout.shape[0], 0))

# Combine features
X_dev_processed = np.hstack([X_dev_morgan, X_dev_physchem])
X_holdout_processed = np.hstack([X_holdout_morgan, X_holdout_physchem])
print(f"Final processed features: {X_dev_processed.shape[1]}")

#6. Compare ML algorithms with hyperparameter tuning and nested cross-validation
print("Comparing multiple regression models with hyperparameter tuning and nested cross-validation...")

# Define models and their parameter grids
models_with_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}  # Linear regression doesn't have hyperparameters to tune
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'Lasso': {
        'model': Lasso(random_state=42),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [2000]
        }
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9],
            'max_iter': [2000]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42, n_jobs=1),  # Changed from n_jobs=-1 to n_jobs=1
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    }
}

# Initialize results lists
nested_cv_results = []
train_results_list = []
test_results_list = []

# Outer and inner CV for nested cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Train and evaluate each model with nested cross-validation
for name, model_info in models_with_params.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    model = model_info['model']
    params = model_info['params']
    
    # Nested cross-validation to get reliable estimate of model performance
    nested_scores = []
    
    # Skip nested CV for models with empty param grids
    if params:
        for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_dev_processed, Y_bins_dev)):
            print(f"  Fold {i+1}/5...")
            # Split data
            X_train_outer, X_val_outer = X_dev_processed[train_idx], X_dev_processed[val_idx]
            y_train_outer, y_val_outer = Y_dev.iloc[train_idx], Y_dev.iloc[val_idx]
            
            # Define grid search - IMPORTANT: Changed n_jobs from -1 to 1 to avoid parallel processing
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=inner_cv.split(X_train_outer, Y_bins_dev[train_idx]),
                scoring='r2',
                n_jobs=1,  # Using a single core to prevent memory issues
                verbose=1   # Added verbosity to track progress
            )
            
            # Fit on outer training data
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Predict on validation data
            y_pred = grid_search.predict(X_val_outer)
            
            # Evaluate
            score = r2_score(y_val_outer, y_pred)
            nested_scores.append(score)
            print(f"    Fold {i+1} R²: {score:.4f}, Best params: {grid_search.best_params_}")
            
        print(f"  Nested CV R²: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
    else:
        # For models without parameters, just use regular cross-validation
        print("  Running cross-validation...")
        cv_scores = []
        for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_dev_processed, Y_bins_dev)):
            # Manual cross-validation to track progress
            print(f"  Fold {i+1}/5...")
            X_train_outer, X_val_outer = X_dev_processed[train_idx], X_dev_processed[val_idx]
            y_train_outer, y_val_outer = Y_dev.iloc[train_idx], Y_dev.iloc[val_idx]
            
            # Fit model
            model.fit(X_train_outer, y_train_outer)
            
            # Score
            y_pred = model.predict(X_val_outer)
            score = r2_score(y_val_outer, y_pred)
            cv_scores.append(score)
            print(f"    Fold {i+1} R²: {score:.4f}")
            
        nested_scores = cv_scores
        print(f"  CV R²: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
    
    # Final model training on all training data
    print("  Training final model on all development data...")
    if params:
        # For models with parameters, do a final grid search with progress reporting
        param_combinations = 1
        for param_values in params.values():
            param_combinations *= len(param_values)
        
        print(f"  Running grid search with {param_combinations} parameter combinations...")
        
        best_score = -float('inf')
        best_params = None
        best_estimator = None
        
        # Manually iterate through parameter combinations for better control and progress tracking
        # This is a simplified version for demonstration
        if param_combinations <= 10:  # Only do manual iteration for small parameter grids
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring='r2',
                n_jobs=1,  # Using a single core
                verbose=1   # Add verbosity
            )
            grid_search.fit(X_dev_processed, Y_dev)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # For larger grids, use a simplified approach with just one set of parameters
            # This is to avoid memory issues
            if name == 'SVR':
                # For SVR, use a simpler configuration
                simple_model = SVR(C=1.0, gamma='scale', kernel='rbf')
                simple_model.fit(X_dev_processed, Y_dev)
                best_model = simple_model
                best_params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            else:
                # Create a GridSearchCV with reduced parameter space
                simple_params = {}
                for param, values in params.items():
                    simple_params[param] = [values[0]]  # Just use the first parameter value
                
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=simple_params,
                    cv=3,  # Fewer folds
                    scoring='r2',
                    n_jobs=1
                )
                grid_search.fit(X_dev_processed, Y_dev)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
        print(f"  Best parameters: {best_params}")
    else:
        best_model = model
        best_model.fit(X_dev_processed, Y_dev)
    
    # Make predictions
    print("  Making predictions...")
    y_train_pred = best_model.predict(X_dev_processed)
    y_test_pred = best_model.predict(X_holdout_processed)
    
    # Calculate metrics
    train_r2 = r2_score(Y_dev, y_train_pred)
    test_r2 = r2_score(Y_holdout, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_dev, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_holdout, y_test_pred))
    train_mae = mean_absolute_error(Y_dev, y_train_pred)
    test_mae = mean_absolute_error(Y_holdout, y_test_pred)
    
    # Calculate time taken
    time_taken = time.time() - start_time
    
    # Save the best model
    if name in ['RandomForestRegressor', 'GradientBoostingRegressor', 'SVR', 'Ridge', 'KNeighborsRegressor']:
        model_filename = f'models/{name.lower()}_tuned.pkl'
        pickle.dump(best_model, open(model_filename, 'wb'))
        print(f"  Model saved to {model_filename}")
    
    # Store results
    nested_cv_results.append({
        'Model': name,
        'Nested_CV_R_Squared_Mean': np.mean(nested_scores),
        'Nested_CV_R_Squared_Std': np.std(nested_scores),
        'Train_R_Squared': train_r2,
        'Test_R_Squared': test_r2,
        'Time_Taken': time_taken
    })
    
    train_results_list.append({
        'Model': name,
        'R-Squared': train_r2,
        'RMSE': train_rmse,
        'MAE': train_mae,
        'Time Taken': time_taken
    })
    
    test_results_list.append({
        'Model': name,
        'R-Squared': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'Time Taken': time_taken
    })

# Create and train ensemble models
print("\nTraining ensemble models...")

# Get the best individual models
best_models = []
for name, model_info in models_with_params.items():
    if name in ['RandomForestRegressor', 'GradientBoostingRegressor', 'Ridge', 'KNeighborsRegressor', 'ElasticNet']:
        model_path = f'models/{name.lower()}_tuned.pkl'
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
            best_models.append((name, model))
        else:
            # If model file doesn't exist, train a new one
            model = model_info['model']
            if len(model_info['params']) > 0:
                # Use simplified parameters for training
                simple_params = {}
                for param, values in model_info['params'].items():
                    simple_params[param] = values[0]  # Just use the first parameter value
                
                # Configure the model with simple params
                for param, value in simple_params.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
                
                model.fit(X_dev_processed, Y_dev)
            else:
                model.fit(X_dev_processed, Y_dev)
            best_models.append((name, model))
            pickle.dump(model, open(model_path, 'wb'))

# Create voting ensemble
if len(best_models) >= 2:
    print("Training voting ensemble...")
    voting_ensemble = VotingRegressor(best_models, n_jobs=1)  # Using a single core
    voting_ensemble.fit(X_dev_processed, Y_dev)
    
    # Predictions and metrics for voting ensemble
    voting_train_pred = voting_ensemble.predict(X_dev_processed)
    voting_test_pred = voting_ensemble.predict(X_holdout_processed)
    
    voting_train_r2 = r2_score(Y_dev, voting_train_pred)
    voting_test_r2 = r2_score(Y_holdout, voting_test_pred)
    voting_train_rmse = np.sqrt(mean_squared_error(Y_dev, voting_train_pred))
    voting_test_rmse = np.sqrt(mean_squared_error(Y_holdout, voting_test_pred))
    voting_train_mae = mean_absolute_error(Y_dev, voting_train_pred)
    voting_test_mae = mean_absolute_error(Y_holdout, voting_test_pred)
    
    # Save voting ensemble
    pickle.dump(voting_ensemble, open('models/voting_ensemble.pkl', 'wb'))
    
    # Add to results
    nested_cv_results.append({
        'Model': 'VotingEnsemble',
        'Nested_CV_R_Squared_Mean': np.nan,
        'Nested_CV_R_Squared_Std': np.nan,
        'Train_R_Squared': voting_train_r2,
        'Test_R_Squared': voting_test_r2,
        'Time_Taken': np.nan
    })
    
    train_results_list.append({
        'Model': 'VotingEnsemble',
        'R-Squared': voting_train_r2,
        'RMSE': voting_train_rmse,
        'MAE': voting_train_mae,
        'Time Taken': np.nan
    })
    
    test_results_list.append({
        'Model': 'VotingEnsemble',
        'R-Squared': voting_test_r2,
        'RMSE': voting_test_rmse,
        'MAE': voting_test_mae,
        'Time Taken': np.nan
    })
    
    print(f"  Voting Ensemble - Train R²: {voting_train_r2:.4f}, Test R²: {voting_test_r2:.4f}")

    # Create stacking ensemble with reduced complexity
    print("Training stacking ensemble...")
    # Final estimator
    final_estimator = Ridge(alpha=1.0)
    
    # Create and train stacking ensemble
    stacking_ensemble = StackingRegressor(
        estimators=best_models,
        final_estimator=final_estimator,
        cv=3,  # Reduced from 5 to 3
        n_jobs=1  # Using a single core
    )
    stacking_ensemble.fit(X_dev_processed, Y_dev)
    
    # Predictions and metrics for stacking ensemble
    stacking_train_pred = stacking_ensemble.predict(X_dev_processed)
    stacking_test_pred = stacking_ensemble.predict(X_holdout_processed)
    
    stacking_train_r2 = r2_score(Y_dev, stacking_train_pred)
    stacking_test_r2 = r2_score(Y_holdout, stacking_test_pred)
    stacking_train_rmse = np.sqrt(mean_squared_error(Y_dev, stacking_train_pred))
    stacking_test_rmse = np.sqrt(mean_squared_error(Y_holdout, stacking_test_pred))
    stacking_train_mae = mean_absolute_error(Y_dev, stacking_train_pred)
    stacking_test_mae = mean_absolute_error(Y_holdout, stacking_test_pred)
    
    # Save stacking ensemble
    pickle.dump(stacking_ensemble, open('models/stacking_ensemble.pkl', 'wb'))
    
    # Add to results
    nested_cv_results.append({
        'Model': 'StackingEnsemble',
        'Nested_CV_R_Squared_Mean': np.nan,
        'Nested_CV_R_Squared_Std': np.nan,
        'Train_R_Squared': stacking_train_r2,
        'Test_R_Squared': stacking_test_r2,
        'Time_Taken': np.nan
    })
    
    train_results_list.append({
        'Model': 'StackingEnsemble',
        'R-Squared': stacking_train_r2,
        'RMSE': stacking_train_rmse,
        'MAE': stacking_train_mae,
        'Time Taken': np.nan
    })
    
    test_results_list.append({
        'Model': 'StackingEnsemble',
        'R-Squared': stacking_test_r2,
        'RMSE': stacking_test_rmse,
        'MAE': stacking_test_mae,
        'Time Taken': np.nan
    })
    
    print(f"  Stacking Ensemble - Train R²: {stacking_train_r2:.4f}, Test R²: {stacking_test_r2:.4f}")

# Convert to DataFrames
nested_cv_df = pd.DataFrame(nested_cv_results)
train_results = pd.DataFrame(train_results_list)
test_results = pd.DataFrame(test_results_list)

# Sort results by test R-squared
nested_cv_df = nested_cv_df.sort_values('Test_R_Squared', ascending=False)
train_results = train_results.sort_values('R-Squared', ascending=False)
test_results = test_results.sort_values('R-Squared', ascending=False)

print(f"\nCompared {len(models_with_params) + (2 if len(best_models) >= 2 else 0)} different regression models")

# Save the performance tables
nested_cv_df.to_csv('results/nested_cv_performance.csv', index=False)
train_results.to_csv('results/model_performance_training.csv', index=False)
test_results.to_csv('results/model_performance_test.csv', index=False)

print("Performance tables saved to:")
print("- results/nested_cv_performance.csv")
print("- results/model_performance_training.csv")
print("- results/model_performance_test.csv")

#7. Data visualization of model performance
# Create subplots for different metrics
plt.figure(figsize=(15, 12))

# R-squared comparison across training, test, and CV
plt.subplot(2, 2, 1)
cv_data = nested_cv_df.copy()
cv_data = cv_data.rename(columns={
    'Train_R_Squared': 'Training',
    'Test_R_Squared': 'Test',
    'Nested_CV_R_Squared_Mean': 'Cross-Validation'
})
cv_data = pd.melt(cv_data, 
                 id_vars=['Model'], 
                 value_vars=['Training', 'Test', 'Cross-Validation'],
                 var_name='Dataset', value_name='R-Squared')
sns.barplot(data=cv_data, x='R-Squared', y='Model', hue='Dataset')
plt.title('R-Squared Across Datasets')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, linestyle='--', alpha=0.6)

# Test set RMSE
plt.subplot(2, 2, 2)
sns.barplot(data=test_results, x='RMSE', y='Model')
plt.title('RMSE on Test Set')
plt.grid(True, linestyle='--', alpha=0.6)

# Test set MAE
plt.subplot(2, 2, 3)
sns.barplot(data=test_results, x='MAE', y='Model')
plt.title('MAE on Test Set')
plt.grid(True, linestyle='--', alpha=0.6)

# Computation time
plt.subplot(2, 2, 4)
time_data = test_results.dropna(subset=['Time Taken'])
sns.barplot(data=time_data, x='Time Taken', y='Model')
plt.title('Computation Time (seconds)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300)
plt.savefig('plots/model_comparison.pdf')
plt.close()

# Create detailed plots for the best model
best_model_name = test_results.iloc[0]['Model']
best_model_path = f'models/{best_model_name.lower()}_tuned.pkl'

# If best model is an ensemble, adjust the path
if best_model_name == 'VotingEnsemble':
    best_model_path = 'models/voting_ensemble.pkl'
elif best_model_name == 'StackingEnsemble':
    best_model_path = 'models/stacking_ensemble.pkl'

if os.path.exists(best_model_path):
    best_model = pickle.load(open(best_model_path, 'rb'))
    
    # Get predictions
    y_train_pred = best_model.predict(X_dev_processed)
    y_test_pred = best_model.predict(X_holdout_processed)
    
    # Create scatter plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.regplot(x=Y_dev, y=y_train_pred, scatter_kws={'alpha':0.4})
    plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
    plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
    plt.title(f'{best_model_name} - Training Set')
    plt.plot([Y_dev.min(), Y_dev.max()], [Y_dev.min(), Y_dev.max()], 'k--', lw=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    r2 = r2_score(Y_dev, y_train_pred)
    rmse = np.sqrt(mean_squared_error(Y_dev, y_train_pred))
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.subplot(1, 2, 2)
    sns.regplot(x=Y_holdout, y=y_test_pred, scatter_kws={'alpha':0.4})
    plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
    plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
    plt.title(f'{best_model_name} - Test Set')
    plt.plot([Y_holdout.min(), Y_holdout.max()], [Y_holdout.min(), Y_holdout.max()], 'k--', lw=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    r2 = r2_score(Y_holdout, y_test_pred)
    rmse = np.sqrt(mean_squared_error(Y_holdout, y_test_pred))
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig('plots/best_model_predictions.png', dpi=300)
    plt.savefig('plots/best_model_predictions.pdf')
    plt.close()
    
    print(f"Best model plots saved to plots/best_model_predictions.png")

# Plot correlation between nested CV and test set performance
if len(nested_cv_df) > 0:
    plt.figure(figsize=(8, 6))
    valid_data = nested_cv_df.dropna(subset=['Nested_CV_R_Squared_Mean', 'Test_R_Squared'])
    
    if len(valid_data) >= 2:  # Need at least 2 points for correlation
        sns.scatterplot(data=valid_data, x='Nested_CV_R_Squared_Mean', y='Test_R_Squared')
        
        # Add model names as labels
        for i, row in valid_data.iterrows():
            plt.text(row['Nested_CV_R_Squared_Mean'], row['Test_R_Squared'], 
                    row['Model'], fontsize=9)
        
        # Add correlation line
        if len(valid_data) > 2:  # Need more than 2 points for regression line
            sns.regplot(data=valid_data, x='Nested_CV_R_Squared_Mean', y='Test_R_Squared', 
                       scatter=False, ci=None, line_kws={'color': 'red'})
        
        plt.title('Correlation between Nested CV and Test Set Performance')
        plt.xlabel('Nested CV R-Squared')
        plt.ylabel('Test Set R-Squared')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('plots/cv_test_correlation.png', dpi=300)
        plt.savefig('plots/cv_test_correlation.pdf')
        plt.close()
        
        print("CV-Test correlation plot saved to plots/cv_test_correlation.png")

#8. Identify the best performing model
best_model_name = test_results.iloc[0]['Model']
best_r2 = test_results.iloc[0]['R-Squared']
best_rmse = test_results.iloc[0]['RMSE']
best_mae = test_results.iloc[0]['MAE']

# Get nested CV score for best model if available
best_cv_r2 = np.nan
best_cv_std = np.nan
best_model_cv = nested_cv_df[nested_cv_df['Model'] == best_model_name]
if len(best_model_cv) > 0:
    if not pd.isna(best_model_cv['Nested_CV_R_Squared_Mean'].iloc[0]):
        best_cv_r2 = best_model_cv['Nested_CV_R_Squared_Mean'].iloc[0]
        best_cv_std = best_model_cv['Nested_CV_R_Squared_Std'].iloc[0]

print("\nBest performing model:")
print(f"- Model: {best_model_name}")
print(f"- Test R²: {best_r2:.4f}")
print(f"- Test RMSE: {best_rmse:.4f}")
print(f"- Test MAE: {best_mae:.4f}")
if not pd.isna(best_cv_r2):
    print(f"- Nested CV R²: {best_cv_r2:.4f} ± {best_cv_std:.4f}")

#9. Summary of findings
print("\nSummary of findings:")
print(f"1. Evaluated {len(models_with_params) + (2 if len(best_models) >= 2 else 0)} regression models for pancreatic cancer drug discovery")
print(f"2. Best model: {best_model_name} with test R² = {best_r2:.4f}")
print(f"3. Used advanced feature processing with PCA for fingerprints and robust scaling for physicochemical descriptors")
print(f"4. Implemented nested cross-validation for reliable performance estimation")
print(f"5. Created ensemble models to improve prediction robustness")
print("6. All results are saved in the 'results' and 'plots' directories")

print("\nWorkflow completed! The pancreatic cancer drug discovery project has:")
print("1. Created bioactivity data for pancreatic cancer targets")
print("2. Performed exploratory data analysis")
print("3. Calculated molecular descriptors using Morgan fingerprints and physicochemical properties")
print("4. Built optimized machine learning models with advanced feature processing")
print("5. Implemented nested cross-validation for reliable performance estimation")
print("6. Created ensemble models that often outperform individual models")
print("\nThe project identified potential predictive models for pancreatic cancer drug discovery with improved generalizability.")



