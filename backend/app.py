from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)

# Model dosyasının doğru yolunu belirlemek
model_path = os.path.join(os.path.dirname(__file__), '../ai/pistachio_model.h5')

# Eğitilmiş modeli yükleyin
model = load_model(model_path)

# Dosya uzantılarını kontrol eden fonksiyon
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Ana sayfa rotası
@app.route('/')
def home():
    return render_template('html/home.html')

# Hakkında sayfası rotası
@app.route('/about')
def about():
    return render_template('html/about.html')

# Fotoğraf yükleme rotası
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'})

        if file and allowed_file(file.filename):
            image = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (128, 128))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            predicted_class = np.round(prediction).astype(int).flatten()[0]

            # Siirt olma olasılığı hesaplanır
            siirt_probability = prediction[0][0] * 100

            # Eğer Siirt tahmin edildiyse bu olasılığı göster
            if predicted_class == 1:
                category = 'Siirt Pistachio'
                probability = siirt_probability
            # Eğer Kırmızı tahmin edildiyse, Siirt olma olasılığını tersine çevir (1 - olasılık)
            else:
                category = 'Kırmızı Pistachio'
                probability = (1 - prediction[0][0]) * 100

            return jsonify({'category': category, 'probability': f'{probability:.2f}%'}), 200

    return render_template('html/upload.html')

if __name__ == '__main__':
    app.run(debug=True)
