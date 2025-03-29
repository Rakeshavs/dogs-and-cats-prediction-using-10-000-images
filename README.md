# 🐶🐱 Dogs vs Cats Classification using CNN

## 📌 Overview

This project is a **Convolutional Neural Network (CNN)** model that classifies images as either a **dog** or a **cat**. The model is trained using **TensorFlow and Keras** on an image dataset and can predict whether an uploaded image contains a dog or a cat.

## 📂 Dataset

The dataset consists of images categorized into two classes:

- **Cats (Label: 0)**
- **Dogs (Label: 1)**

📁 **Dataset Structure:**

```
/dataset/
    /training_set/
        /cats/
        /dogs/
    /test_set/
        /cats/
        /dogs/
```

## 🚀 Installation

Before running the project, install the necessary dependencies:

```bash
pip install tensorflow keras numpy matplotlib
```

## 🏗 Model Architecture

The CNN model consists of the following layers:

- **Conv2D + ReLU** (Feature Extraction)
- **MaxPooling2D** (Downsampling)
- **Flatten** (Convert to a 1D array)
- **Dense (Fully Connected Layers)**
- **Sigmoid Activation** (Binary Classification)

## 📌 Training the Model

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=20)
```

## 📸 Making Predictions

To predict whether an image is a dog or a cat:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('test_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("The model predicts:", prediction)
```

## 💾 Saving and Loading the Model

To save the trained model:

```python
cnn.save("cnn_model.keras")
```

To load the saved model:

```python
from tensorflow.keras.models import load_model
cnn = load_model("cnn_model.keras")
```

## 📊 Performance Metrics

The model achieves:
✔ **Accuracy: \~85-90%** (Can vary based on dataset size and training epochs)
✔ **Loss: Optimized using Adam optimizer**

## 🛠 Future Improvements

- 🔹 Use a larger dataset for better accuracy
- 🔹 Implement **Transfer Learning** (e.g., VGG16, ResNet)
- 🔹 Deploy as a **web app using Flask or FastAPI**

## 📜 License

This project is open-source and free to use. Feel free to contribute!

---

🔗 **Author:** Rakesh Medhari\
🚀 Happy Coding! 🎯

