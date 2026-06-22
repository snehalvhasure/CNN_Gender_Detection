import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

model = tf.keras.Sequential()
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Input = tf.keras.Input
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

# --------------------------------------------------------------------
# DATASET PATHS

path_female = "D:/AI Course (All DataSet)/DataSetsGender/female/"
path_male = "D:/AI Course (All DataSet)/DataSetsGender/male/"

# --------------------------------------------------------------------
# LOAD IMAGE NAMES

female_img_names = os.listdir(path_female)
male_image_names = os.listdir(path_male)

# --------------------------------------------------------------------
# CREATE DATASET AND LABELS
# Female = 0
# Male = 1
# --------------------------------------------------------------------

dataset = []
label = []

# --------------------------
# Female Images
# --------------------------

for name in female_img_names:
    f_image = cv2.imread(path_female + name)

    if f_image is None:
        continue

    f_image = cv2.resize(f_image, (100, 150))

    dataset.append(f_image)
    label.append(0)

# --------------------------
# Male Images
# --------------------------

for name in male_image_names:
    male_img = cv2.imread(path_male + name)

    if male_img is None:
        continue

    male_img = cv2.resize(male_img, (100, 150))

    dataset.append(male_img)
    label.append(1)

# --------------------------------------------------------------------
# DATASET INFORMATION

print("Total Images :", len(dataset))
print("Total Labels :", len(label))

print("\nClass Distribution:\n")
print(pd.DataFrame(label).value_counts())

# --------------------------------------------------------------------
# CONVERT LISTS TO NUMPY ARRAYS


ary_dataset = np.array(dataset)
ary_label = np.array(label)

print("\nDataset Shape :", ary_dataset.shape)
print("Label Shape   :", ary_label.shape)

# --------------------------------------------------------------------
# IMAGE SCALING

ary_dataset_scale = ary_dataset / 255.0

# --------------------------------------------------------------------
# TRAIN TEST SPLIT

X = ary_dataset_scale.copy()
y = ary_label

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=2
)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

print("Training Labels Shape :", y_train.shape)
print("Testing Labels Shape  :", y_test.shape)

# --------------------------------------------------------------------
# BUILD CNN MODEL

cnn_model = model

cnn_model.add(
    Input(shape=(150, 100, 3))
)

# --------------------------
# Convolution Block 1
# --------------------------

cnn_model.add(
    Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        activation='relu',
        padding='valid'
    )
)

cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='valid'
    )
)

# --------------------------
# Convolution Block 2
# --------------------------

cnn_model.add(
    Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        activation='relu',
        padding='valid'
    )
)

cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='valid'
    )
)

# --------------------------
# Convolution Block 3
# --------------------------

cnn_model.add(
    Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        activation='relu',
        padding='valid'
    )
)

cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        padding='valid'
    )
)

# --------------------------------------------------------------------
# FULLY CONNECTED LAYERS

cnn_model.add(Flatten())

cnn_model.add(
    Dense(
        512,
        activation='relu'
    )
)

cnn_model.add(
    Dense(
        1024,
        activation='relu'
    )
)

cnn_model.add(
    Dense(
        512,
        activation='relu'
    )
)

cnn_model.add(
    Dense(
        2,
        activation='softmax'
    )
)

# MODEL SUMMARY

cnn_model.summary()

# COMPILE MODEL

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL

print("\nTraining Started...\n")

history = cnn_model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.1
)

# EVALUATE MODEL

print("\nEvaluating Model...\n")

loss, accuracy = cnn_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==============================")
print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)
print("==============================")

# PLOT ACCURACY GRAPH

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Gender Detection Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# PREDICTIONS

y_pred = cnn_model.predict(X_test)

predicted_labels = np.argmax(
    y_pred,
    axis=1
)

print("\nFirst 10 Predictions:")
print(predicted_labels[:10])

print("\nActual Labels:")
print(y_test[:10])

# SAVE MODEL

cnn_model.save("cnn_gender_detection_model.h5")
print("\nModel Saved Successfully!")