import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

# Wczytywanie danych
cifar10 = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizacja wartości pikseli 
X_train, X_test = X_train / 255.0, X_test / 255.0

# Wyświetlenie przykładowych obrazów z datasetu
class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń', 'pies', 'żaba', "koń", "statek", 'ciężarówka']

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axes[i].imshow(X_train[i])
    axes[i].set_title(class_names[y_train[i] [0]])
    axes[i].axis('off')
plt.show()

# Tworzymy sieć neuronową 
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)), # Warstwa konwolucyjna 1
    keras.layers.MaxPooling2D((2,2)), # Warstwa pooling 1
    keras.layers.Conv2D(64, (3,3), activation='relu'), # Warstwa konwolucyjna 2
    keras.layers.MaxPooling2D((2,2)), # Warstwa pooling 2
    keras.layers.Conv2D(64, (3,3), activation='relu'), # Warstwa konwolucyjna 3
    keras.layers.Flatten(), # Spłasczenie map cech
    keras.layers.Dense(64, activation='relu'), # Warstwa ukryta w pełni połączona 64 neurony 
    keras.layers.Dense(10, activation='softmax') # Warstwa wyjściowa (10 klas)
])

# Kompilacja modelu 
model.compile(
    optimizer ='adam',
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Wyświetlenie architektury modelu
model.summary()

# Trenowanie modelu na zbiorze MNIST
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
print(history)

# Dokładność modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Dokładność na zbiorze testowym to {test_acc:.4f}")

model.save("model_cnn.h5")
print("Model został zapisany jako model_cnn.h5")

# Wybieramy losowy obrazek z testowego zbioru
idx = np.random.randint(len(X_test))
img = X_test[idx]

prediction = model.predict(img)
predicted_label = np.argmax(prediction)

# Wyświetlenie obrazka i przewidywanej klasy
plt.imshow(img, cmap='gray')
plt.title(f'przewidywana klasa: {predicted_label}')
plt.show()