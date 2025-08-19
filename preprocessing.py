from sklearn.preprocessing import LabelEncoder

def encode_labels(df):
    le_dict = {}
    for col in ["masterCategory", "subCategory", "gender"]:  # chỉ encode các cột thật sự tồn tại
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    return df, le_dict
