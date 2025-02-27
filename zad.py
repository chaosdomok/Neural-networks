import cv2
import numpy as np 
from tensorflow import keras

loaded_model = keras.models.load_model("model_cnn.h5")
print("Model został załadowany")

img = cv2.imread("images/cats.jpg")
img = cv2.resize(img, (32, 32))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = loaded_model.predict(img)
predicted_label = np.argmax(prediction)

class_names = ['samolot', 'samochód', 'ptak', 'kot', 'jeleń', 'pies', 'żaba', "koń", "statek", 'ciężarówka']

print(f"Model przewiduje: {class_names[predicted_label]}")
