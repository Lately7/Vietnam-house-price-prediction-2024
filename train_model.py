import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("vietnam_housing_cleaned_encoded.csv")

print("Shape dataset:", df.shape)
print(df.head())

# =========================
# 2. CHỌN X, y
# =========================
# Dùng Price_log làm target để model ổn định hơn
target_col = "Price_log"

drop_cols = ["Price", "Price_log"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[target_col]

print("\nSố feature:", len(feature_cols))
print("Tên target:", target_col)

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 4. HÀM ĐÁNH GIÁ
# =========================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    model.fit(X_train, y_train)

    # predict trên log-scale
    y_pred_log = model.predict(X_test)

    # predict trên scale gốc để dễ hiểu hơn
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"\n===== {model_name} =====")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    return {
        "name": model_name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

# =========================
# 5. KHAI BÁO MODEL
# =========================
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )),
    ("Gradient Boosting", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ))
]

# =========================
# 6. TRAIN + EVALUATE
# =========================
results = []

for name, model in models:
    result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    results.append(result)

# =========================
# 7. CHỌN MODEL TỐT NHẤT
# =========================
# Ưu tiên RMSE thấp nhất
best_result = min(results, key=lambda x: x["rmse"])
best_model = best_result["model"]

print("\n==============================")
print("BEST MODEL:", best_result["name"])
print("BEST RMSE :", round(best_result["rmse"], 4))
print("BEST MAE  :", round(best_result["mae"], 4))
print("BEST R2   :", round(best_result["r2"], 4))
print("==============================")

# =========================
# 8. FEATURE IMPORTANCE
# =========================
if hasattr(best_model, "feature_importances_"):
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 15 feature importance:")
    print(importances.head(15))

    importances.to_csv("feature_importance.csv", index=False)

# =========================
# 9. LƯU MODEL
# =========================
joblib.dump(best_model, "best_house_price_model.pkl")
joblib.dump(feature_cols, "model_features.pkl")

print("\nĐã lưu:")
print("- best_house_price_model.pkl")
print("- model_features.pkl")
if "importances" in locals():
    print("- feature_importance.csv")