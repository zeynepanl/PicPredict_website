import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Test verilerini yükleme
data = np.load('ai/test_data.npz')
X_test = data['X_test']
y_test = data['y_test']

# Eğitilmiş modeli yükleyin
model = load_model('ai/pistachio_model.h5')

# Modeli kullanarak tahmin yapın
predictions = model.predict(X_test)

# Tahminleri sınıflandırın (binary classification için 0 ve 1 olarak)
predicted_classes = np.round(predictions).astype(int).flatten()

# Tahmin sonuçlarını ve doğruluğunu yazdırın
correct_predictions = 0
incorrect_predictions = 0

for i, (true_label, pred) in enumerate(zip(y_test, predicted_classes)):
    is_correct = "Doğru" if true_label == pred else "Yanlış"
    if true_label == pred:
        correct_predictions += 1
    else:
        incorrect_predictions += 1
    print(f"Index: {i}, Predicted Class: {pred}, True Class: {true_label}, Tahmin: {is_correct}")

# Doğru ve yanlış tahmin sayılarını yazdırın
print(f"Doğru Tahmin Sayısı: {correct_predictions}")
print(f"Yanlış Tahmin Sayısı: {incorrect_predictions}")