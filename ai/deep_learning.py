import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Veri seti yolunun tanımlanması
data_dir = r'C:\Users\zeynep\Desktop\Pistachio_Image_Dataset'

# Görüntü dosyalarının ve etiketlerin yüklenmesi
def load_images_and_labels(data_dir):
    images, labels = [], []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(img_path)
                    labels.append(class_name)
    return images, labels

image_paths, labels = load_images_and_labels(data_dir)

# Pandas DataFrame'e görüntü dosya yolları ve etiketlerin eklenmesi
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# Görüntülerin yüklenmesi ve işlenmesi
def process_image(image_path, size=(128, 128)):  # Boyutu küçülttüm
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image = image / 255.0
    return image

df['processed_image'] = df['image_path'].map(lambda x: process_image(x, size=(128, 128)))  # Boyutu küçülttüm

# Veri setinin eğitim, doğrulama ve test setlerine 70/20/10 oranında bölünmesi
images = np.array([img for img in df['processed_image'].values])
labels = pd.factorize(df['label'])[0]

# Eğitim seti için %70
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)

# Kalan %30'u doğrulama ve test setlerine %20/%10 olarak bölme
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3333, random_state=42)

# Eğitim, doğrulama ve test setlerinin boyutlarını yazdırma
print(f"Eğitim veri seti boyutu: {X_train.shape}")
print(f"Doğrulama veri seti boyutu: {X_val.shape}")
print(f"Test veri seti boyutu: {X_test.shape}")

# Test verilerini kaydetme
np.savez(os.path.join(os.path.dirname(__file__), 'test_data.npz'), X_test=X_test, y_test=y_test)

# Data Augmentation tekniklerinin uygulanması
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.1
)

train_generator = datagen.flow(X_train, y_train, batch_size=16)  # Batch size'ı düşürdüm
validation_generator = datagen.flow(X_val, y_val, batch_size=16)  # Batch size'ı düşürdüm

# VGG16 modelini yükleyin, son fully connected katmanları dahil olmadan
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))  # Boyutu küçülttüm

# Modelin katmanlarını dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modelin üzerine ekleme yapın
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma için sigmoid aktivasyon fonksiyonu kullanılır
])

# Modeli derleyin
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli eğitin
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 16,  # Batch size'a göre ayarlandı
    validation_steps=len(X_val) // 16  # Batch size'a göre ayarlandı
)

# Modeli değerlendirin
test_generator = datagen.flow(X_test, y_test, batch_size=16)  # Batch size'ı düşürdüm
test_loss, test_acc = model.evaluate(test_generator, steps=len(X_test) // 16)  # Batch size'a göre ayarlandı
print(f"Test Doğruluğu: {test_acc:.4f}")

# Fine-Tuning (İnce Ayar) için son katmanları açma
for layer in base_model.layers[-4:]:  # Son 4 katmanı açalım
    layer.trainable = True

# Modeli yeniden derleyin
model.compile(optimizer=Adam(learning_rate=0.00001),  # Öğrenme oranını daha küçük bir değer yapın
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli ince ayar yaparak yeniden eğitin
history_fine = model.fit(
    train_generator,
    epochs=10,  # Daha az epoch ile ince ayar yapın
    validation_data=validation_generator,
    steps_per_epoch=len(X_train) // 16,
    validation_steps=len(X_val) // 16
)

# Fine-tuning sonrası modeli değerlendirin
test_loss_fine, test_acc_fine = model.evaluate(test_generator, steps=len(X_test) // 16)
print(f"Fine tuning Sonrası Test Doğruluğu: {test_acc_fine:.4f}")

# Fine-tuning sonrası modeli kaydedin
model.save(os.path.join(os.path.dirname(__file__), 'pistachio_model.h5'))

# Tahminleri al
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Eğitim ve doğrulama grafikleri
total_epochs = list(range(len(history.history['accuracy']) + len(history_fine.history['accuracy'])))
accuracy = history.history['accuracy'] + history_fine.history['accuracy']
val_accuracy = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(total_epochs, accuracy, label='Eğitim Doğruluğu')
plt.plot(total_epochs, val_accuracy, label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(total_epochs, loss, label='Eğitim Kaybı')
plt.plot(total_epochs, val_loss, label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')

plt.show()

# İşlenmiş verilerden 3'er örnek gösterme fonksiyonu
def show_processed_images(df, n=3):
    classes = df['label'].unique()
    for cls in classes:
        class_samples = df[df['label'] == cls].sample(n)
        for index, row in class_samples.iterrows():
            image = row['processed_image']
            label = row['label']
            plt.imshow(image)
            plt.title(f"Label: {label}")
            plt.axis('off')
            plt.show()

# Her sınıftan 3 işlenmiş görüntü göster
show_processed_images(df, n=3)
