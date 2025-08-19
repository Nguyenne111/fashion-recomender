import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# 1. Load CSV
# ==========================
csv_path = r"C:\Users\Admin\Desktop\fashion-recomender\data\styles.csv"
df = pd.read_csv(csv_path, on_bad_lines='skip')

# Thêm cột filename = id.jpg
df['filename'] = df['id'].astype(str) + ".jpg"

# ==========================
# 2. Kiểm tra thư mục ảnh
# ==========================
# ⚠️ Chỉnh lại đường dẫn này đúng nơi chứa ảnh của bạn
image_dir = r"C:\Users\Admin\Desktop\fashion-recomender\data\images"

# Giữ lại ảnh thực sự tồn tại
df['filepath'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
df = df[df['filepath'].apply(os.path.exists)]  # chỉ giữ file tồn tại

print("👉 Tổng số ảnh trong CSV:", len(df))
print("👉 Số ảnh hợp lệ (tồn tại trên disk):", len(df))
print("👉 Một vài id:", df["id"].head().tolist())

# ==========================
# 3. Test thử 1 ảnh
# ==========================
if len(df) > 0:
    sample_path = os.path.join(image_dir, str(df["id"].iloc[0]) + ".jpg")
    print("👉 Test đường dẫn ảnh:", sample_path, "=>", os.path.exists(sample_path))
else:
    print("❌ Không có ảnh nào hợp lệ, cần kiểm tra lại thư mục images!")

# ==========================
# 4. Chia train/val
# ==========================
if len(df) > 0:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # ==========================
    # 5. Tạo generator
    # ==========================
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        train_df,
        directory=image_dir,
        x_col="filename",
        y_col="masterCategory",   # hoặc "subCategory" tùy bạn train
        target_size=(128,128),
        class_mode="categorical"
    )

    val_generator = datagen.flow_from_dataframe(
        val_df,
        directory=image_dir,
        x_col="filename",
        y_col="masterCategory",
        target_size=(128,128),
        class_mode="categorical"
    )
