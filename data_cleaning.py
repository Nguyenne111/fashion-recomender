import os
import pandas as pd

def load_and_clean():
    # Lấy thư mục gốc của project (cha của src)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "data", "styles.csv")

    print(f"👉 Đang load file: {path}")  # debug cho chắc
    df = pd.read_csv(path, on_bad_lines="skip")

    # Xử lý cơ bản
    df = df.dropna(subset=["id", "productDisplayName"])
    return df
