import pandas as pd
import numpy as np

# Đọc file
df = pd.read_csv("vietnam_housing_dataset.csv")

# Chuẩn hóa tên cột
df.columns = [c.strip() for c in df.columns]

rename_map = {
    "Address": "Address",
    "Area": "Area",
    "Frontage": "Frontage",
    "Access Road": "Access_Road",
    "House direction": "House_direction",
    "Balcony direction": "Balcony_direction",
    "Floors": "Floors",
    "Bedrooms": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Legal status": "Legal_status",
    "Furniture state": "Furniture_state",
    "Price": "Price"
}
df = df.rename(columns=rename_map)


text_cols = ["Address", "House_direction", "Balcony_direction", "Legal_status", "Furniture_state"]

for col in text_cols:
    if col in df.columns:
        #Đưa data về string
        df[col] = df[col].astype(str).str.strip()
        #Loại bỏ những giá trị rỗng như NaN, None, "", "  "
        df[col] = df[col].replace({
            "nan": np.nan,
            "None": np.nan,
            "": np.nan,
            "  ": np.nan
        })


#Chuyển data về numeric
num_cols = ["Area", "Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms", "Price"]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Giá và diện tích phải > 0
df = df[(df["Price"].notna()) & (df["Price"] > 0)]
df = df[(df["Area"].notna()) & (df["Area"] > 0)]

#Loại bỏ những trường hợp làm lệch mô hình
df = df[df["Area"] <= 1000]
#Loại bỏ giá trị cực đoan (0.5% top đầu) gây lệch mô hình
df = df[df["Price"] <= df["Price"].quantile(0.995)]  


# Xóa duplicate toàn bộ
df = df.drop_duplicates()

# Xóa duplicate nếu như có cùng Address, Arena, Price (ví dụ như 2 nhà cùng 1 chỗ nhưng giá nhà lại khác nhau, nghĩa là nó vô lý)
subset_dup = ["Address", "Area", "Price"]
subset_dup = [c for c in subset_dup if c in df.columns]
df = df.drop_duplicates(subset=subset_dup, keep="first")

#Loại bỏ cột thiếu quá nhiều dữ liệu làm lệch mô hình
missing_ratio = df.isna().mean().sort_values(ascending=False)
print("\nTỉ lệ dữ liệu bị thiếu:")
print(missing_ratio)

drop_cols = []
for col in ["Balcony_direction", "House_direction"]:
    #Nếu cột có data bị thiếu > 65% dữ liệu thì loại bỏ
    if col in df.columns and df[col].isna().mean() > 0.65:
        drop_cols.append(col)

#Bỏ cột address vì chuỗi quá phức tạp, là địa chỉ tùy ý nên ko thể xử lý riêng, phải tích hợp thêm NLP
if "Address" in df.columns:
    drop_cols.append("Address")

df = df.drop(columns=drop_cols, errors="ignore")

print("\nCác cột đã bỏ:", drop_cols)


#Tạo những cột missing cho những cột bị thiếu dữ liệu
candidate_missing_flag_cols = ["Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms"]

for col in candidate_missing_flag_cols:
    if col in df.columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)


# Điền dữ liệu bị thiếu bằng những dữ liệu thay thế
numeric_cols_now = df.select_dtypes(include=[np.number]).columns.tolist()
# Loại bỏ cột Price (cột chứa số) vì đây là cột output, cột kết quả nên không được điền lung tung
numeric_feature_cols = [c for c in numeric_cols_now if c != "Price"]

# Numeric -> median, thay thế dữ liệu số thiếu bằng median của cột đó
for col in numeric_feature_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

cat_cols_now = df.select_dtypes(include=["object"]).columns.tolist()
# Categorical -> Unknown, thay thế dữ liệu dạng chuỗi bị thiếu bằng "Unknown"
for col in cat_cols_now:
    df[col] = df[col].fillna("Unknown")


# Loại bỏ những giá trị gây lệch mô hình quá lớn quả nhỏ (outlier) bằng IQR
def clip_outliers_iqr(dataframe, cols, k=1.5):
    df_out = dataframe.copy()
    for col in cols:
        if col in df_out.columns:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k * iqr
            df_out[col] = df_out[col].clip(lower=low, upper=high)
    return df_out

outlier_cols = ["Area", "Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms"]
outlier_cols = [c for c in outlier_cols if c in df.columns]

df = clip_outliers_iqr(df, outlier_cols, k=1.5)


# (optional) Tạo ra những giá trị bổ sung để mô hình học tốt hơn
# - Ví dụ: Cùng diện tích 
# + Ít phòng -> rộng -> xịn -> giá cao
# + Nhiều phòng -> chật -> giá thấp
if "Area" in df.columns and "Bedrooms" in df.columns:
    df["Area_per_Bedroom"] = df["Area"] / (df["Bedrooms"] + 1)

if "Area" in df.columns and "Bathrooms" in df.columns:
    df["Area_per_Bathroom"] = df["Area"] / (df["Bathrooms"] + 1)

if "Frontage" in df.columns and "Area" in df.columns:
    df["Frontage_Area_ratio"] = df["Frontage"] / (df["Area"] + 1)


# Biến chữ thành số
cat_cols_now = df.select_dtypes(include=["object"]).columns.tolist()
df_clean = pd.get_dummies(df, columns=cat_cols_now, drop_first=True)


# Biến Price thành log, vì giá nhà thường lệch phải mạnh -> log giúp model ổn hơn
df_clean["Price_log"] = np.log1p(df_clean["Price"])


df.to_csv("vietnam_housing_cleaned_raw.csv", index=False)
df_clean.to_csv("vietnam_housing_cleaned_encoded.csv", index=False)

print("\nShape sau clean raw:", df.shape)
print("Shape sau encode:", df_clean.shape)
print("\nThiếu dữ liệu còn lại:")
print(df_clean.isna().sum().sort_values(ascending=False).head(20))

print("\nĐã lưu:")
print("- vietnam_housing_cleaned_raw.csv")
print("- vietnam_housing_cleaned_encoded.csv")