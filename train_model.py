import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Define the model
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(24,24,1)))
model.add(MaxPooling2D((1,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((1,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((1,1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load data
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (24, 24))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

open_images, open_labels = load_images_from_folder('Trained_Open_Eyes', 1)
close_images, close_labels = load_images_from_folder('Trained_Close_ eyes_Images', 0)

X = np.array(open_images + close_images)
y = np.array(open_labels + close_labels)

# Convert labels to categorical
from keras.utils import to_categorical
y = to_categorical(y, num_classes=2)

X = X.reshape(X.shape[0], 24, 24, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('models/custmodel.h5')

print("Model trained and saved.")