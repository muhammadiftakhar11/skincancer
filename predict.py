import tensorflow as tf
import cv2
import numpy as np

# trained model load
model = tf.keras.models.load_model("skin_model.h5")

# image read
img = cv2.imread("test.jpg")

# resize image
img = cv2.resize(img, (224,224))

# normalize
img = img/255.0

# reshape
img = np.reshape(img, (1,224,224,3))

# prediction
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Result: Malignant (Skin Cancer)")
else:
    print("Result: Benign (Normal Skin)")