import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from data_cleaning import load_and_clean
from preprocessing import encode_labels, load_images

if __name__ == "__main__":
    # Load dữ liệu và model
    df = load_and_clean()
    df, le_dict = encode_labels(df)
    X = load_images(df)
    y_true = df['category'].values
    
    model = load_model("model/fashion_model.h5")
    y_pred = np.argmax(model.predict(X), axis=1)
    
    print(classification_report(y_true, y_pred))
