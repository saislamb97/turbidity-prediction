import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For better visualization in EDA (optional)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================
df = pd.read_csv("dataset(combined).csv")

# Ensure data is numeric and drop rows with invalid data
df["Voltage"] = pd.to_numeric(df["Voltage"], errors="coerce")
df["Second"] = pd.to_numeric(df["Second"], errors="coerce")
df["Turbidity"] = pd.to_numeric(df["Turbidity"], errors="coerce")
df = df.dropna(subset=["Voltage", "Second", "Turbidity"])

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("Descriptive Statistics:\n", df.describe())

# Histogram plots for each variable
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.hist(df["Voltage"], bins=20, edgecolor='k')
plt.title("Voltage Distribution")
plt.xlabel("Voltage (V)")
plt.ylabel("Count")

plt.subplot(1, 3, 2)
plt.hist(df["Second"], bins=20, edgecolor='k')
plt.title("Time Distribution")
plt.xlabel("Second (s)")

plt.subplot(1, 3, 3)
plt.hist(df["Turbidity"], bins=20, edgecolor='k')
plt.title("Turbidity Distribution")
plt.xlabel("Turbidity (NTU)")
plt.tight_layout()
plt.savefig("eda_histograms.png")
plt.close()
print("Saved histogram plots as 'eda_histograms.png'")

# Scatter plot: Time vs Turbidity colored by Voltage
plt.figure(figsize=(6, 4))
scatter = plt.scatter(df["Second"], df["Turbidity"], c=df["Voltage"], cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Voltage (V)")
plt.xlabel("Time (s)")
plt.ylabel("Turbidity (NTU)")
plt.title("Time vs Turbidity (colored by Voltage)")
plt.tight_layout()
plt.savefig("time_vs_turbidity.png")
plt.close()
print("Saved scatter plot 'time_vs_turbidity.png'")

# ============================================================
# 3. RUN SEGMENTATION: New run when Second == 2
# ============================================================
df['run'] = (df['Second'] == 2).cumsum()
df = df.sort_values(["run", "Second"]).reset_index(drop=True)
num_runs = df['run'].nunique()
print("Total number of runs:", num_runs)

# Optional: save the dataset with run info
df.to_csv("dataset_droppednull_with_runs.csv", index=False)
print("Exported dataset with run info to 'dataset_droppednull_with_runs.csv'")

# ============================================================
# 4. RUN-LEVEL ANALYSIS
#    A) Turbidity near 5 minutes (closest to 300 s)
#    B) Fastest time to achieve lowest turbidity per run
# ============================================================
turb_at_5min = []
lowest_turb_results = []

for run_id, run_df in df.groupby("run"):
    # A) Analysis for turbidity near 300 seconds
    if run_df["Second"].max() < 300:
        continue
    # Get row closest to 300 s
    idx = (run_df["Second"] - 300).abs().idxmin()
    row = run_df.loc[idx]
    turb_at_5min.append({
        "run": run_id,
        "Voltage": row["Voltage"],
        "Second": row["Second"],
        "Turbidity_5min": row["Turbidity"]
    })
    
    # B) Fastest time to achieve minimum turbidity in this run
    run_df = run_df.dropna(subset=["Turbidity"])
    if run_df.empty:
        continue
    min_turb = run_df["Turbidity"].min()
    earliest_time = run_df[run_df["Turbidity"] == min_turb]["Second"].min()
    voltage_run = run_df.iloc[0]["Voltage"]  # assuming constant voltage within a run
    lowest_turb_results.append({
        "run": run_id,
        "Voltage": voltage_run,
        "MinTurbidity": min_turb,
        "EarliestTime_s": earliest_time
    })

turb_5min_df = pd.DataFrame(turb_at_5min).dropna(subset=["Turbidity_5min"])
lowest_turb_df = pd.DataFrame(lowest_turb_results)

print("\n--- Turbidity near 5 minutes (≈300 s) per run ---")
print(turb_5min_df)
if not turb_5min_df.empty:
    min_5min = turb_5min_df["Turbidity_5min"].min()
    best_runs = turb_5min_df[turb_5min_df["Turbidity_5min"] == min_5min]
    print(f"\nLowest turbidity near 300 s = {min_5min:.2f} NTU, achieved in run(s):")
    print(best_runs[["run", "Voltage", "Second"]])
else:
    print("No runs extend near 300 seconds; cannot determine 5-min turbidity.")

print("\n--- Fastest Time to Achieve Lowest Turbidity (per run) ---")
print(lowest_turb_df)
if not lowest_turb_df.empty:
    global_min = lowest_turb_df["MinTurbidity"].min()
    best_runs_global = lowest_turb_df[lowest_turb_df["MinTurbidity"] == global_min]
    print(f"\nOverall lowest turbidity = {global_min:.2f} NTU, achieved in run(s):")
    print(best_runs_global[["run", "Voltage", "EarliestTime_s"]])
    fastest = best_runs_global["EarliestTime_s"].min()
    row_fastest = best_runs_global[best_runs_global["EarliestTime_s"] == fastest].iloc[0]
    print(f"Fastest time to achieve {global_min:.2f} NTU is {fastest} s in run {row_fastest['run']} at Voltage {row_fastest['Voltage']}")
else:
    print("No data to determine the fastest time for lowest turbidity.")

# Plot bar charts for the above analyses
if not turb_5min_df.empty:
    plt.figure(figsize=(6, 4))
    plt.bar(turb_5min_df["run"].astype(str), turb_5min_df["Turbidity_5min"])
    plt.title("Turbidity near 300 s by Run")
    plt.xlabel("Run")
    plt.ylabel("Turbidity (NTU)")
    plt.tight_layout()
    plt.savefig("turbidity_5min_by_run.png")
    plt.close()
    print("Saved figure: 'turbidity_5min_by_run.png'")

if not lowest_turb_df.empty:
    plt.figure(figsize=(6, 4))
    plt.bar(lowest_turb_df["run"].astype(str), lowest_turb_df["EarliestTime_s"])
    plt.title("Fastest Time to Lowest Turbidity by Run")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("fastest_time_to_min_turbidity_by_run.png")
    plt.close()
    print("Saved figure: 'fastest_time_to_min_turbidity_by_run.png'")

# ============================================================
# 5. MACHINE LEARNING: PREDICTING TURBIDITY
#    We enhance the model by:
#       - Adding new features to capture non-linear effects.
#       - Comparing a baseline Linear Regression and a tuned Random Forest.
#       - Reporting R2, RMSE, and MAE on the test set.
# ============================================================
# Create new features for non-linearity
df_ml = df.copy()
df_ml['Second_squared'] = df_ml['Second'] ** 2
df_ml['Voltage_squared'] = df_ml['Voltage'] ** 2
df_ml['Voltage_x_Second'] = df_ml['Voltage'] * df_ml['Second']

# Define feature set and target
features = ["Voltage", "Second", "Second_squared", "Voltage_squared", "Voltage_x_Second"]
X = df_ml[features]
y = df_ml["Turbidity"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Baseline Linear Regression
lin_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print("\n--- Baseline Linear Regression ---")
print(f"R²: {r2_lr:.3f}, RMSE: {rmse_lr:.3f}, MAE: {mae_lr:.3f}")

# Random Forest with Hyperparameter Tuning using GridSearchCV
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=1))
])
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("\n--- Tuned Random Forest Regression ---")
print("Best Parameters:", grid_search.best_params_)
print(f"R²: {r2_rf:.3f}, RMSE: {rmse_rf:.3f}, MAE: {mae_rf:.3f}")

# Cross-Validation Scores for Random Forest (using the best model)
cv_scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print(f"5-Fold CV R² Scores (RF): {cv_scores_rf}")
print(f"Mean CV R² Score (RF): {cv_scores_rf.mean():.3f}")

# Plot predicted vs actual for the best RF model
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_rf, alpha=0.7, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Turbidity")
plt.ylabel("Predicted Turbidity")
plt.title("Predicted vs Actual Turbidity (RF)")
plt.tight_layout()
plt.savefig("predicted_vs_actual_rf.png")
plt.close()
print("Saved figure: 'predicted_vs_actual_rf.png'")

# Feature Importances from Random Forest
rf_model = best_rf.named_steps['rf']
importances = rf_model.feature_importances_
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(5, 3))
plt.barh(feat_importance_df["Feature"], feat_importance_df["Importance"])
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png")
plt.close()
print("Saved figure: 'rf_feature_importances.png'")

# ============================================================
# 6. FINAL SUMMARY
# ============================================================
print("\n============ FINAL SUMMARY ============")
print(f"Total number of runs: {num_runs}\n")

# Run-level insights
if not turb_5min_df.empty:
    print(f"(1) Lowest Turbidity near 300 s = {min_5min:.2f} NTU, achieved in run(s):")
    print(best_runs[["run", "Voltage", "Second"]])
else:
    print("(1) No runs extend near 300 s; cannot determine 5-min turbidity.")

if not lowest_turb_df.empty:
    print(f"\n(2) Overall Lowest Turbidity across runs = {global_min:.2f} NTU")
    print("Achieved in run(s):")
    print(best_runs_global[["run", "Voltage", "EarliestTime_s"]])
    print(f"Fastest time: {fastest} s in run {row_fastest['run']} at Voltage {row_fastest['Voltage']}")
else:
    print("(2) No data to determine overall lowest turbidity.")

# Model comparison
print("\n(3) MODEL PERFORMANCE:")
print("Baseline Linear Regression:")
print(f"  R²: {r2_lr:.3f}, RMSE: {rmse_lr:.3f}, MAE: {mae_lr:.3f}")
print("Tuned Random Forest Regression:")
print(f"  R²: {r2_rf:.3f}, RMSE: {rmse_rf:.3f}, MAE: {mae_rf:.3f}")
print("========================================\n")
