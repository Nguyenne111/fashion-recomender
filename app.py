from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2, numpy as np
import os

app = Flask(__name__)
model = load_model('model/fashion_model.h5')
IMAGE_SIZE = (128,128)

def prepare_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_array = prepare_image(file)
            pred_class = np.argmax(model.predict(img_array), axis=1)[0]
            prediction = f"Predicted category: {pred_class}"
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
