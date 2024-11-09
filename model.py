import os 
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping 
from skimage.feature import hog
import pickle
image_folder = 'head_ct'
labels_df = pd.read_csv('labels.csv')
images = []
labels = []
image_filenames = []
for index, row in labels_df.iterrows():
    img_id = str(row['id']).zfill(3)
    img_path = os.path.join(image_folder, f"{img_id}.png")
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(row['hemorrhage']) 
        image_filenames.append(f"{img_id}.png")
    else:
        print(f"Image not found: {img_path}")
X = np.array(images)
y = np.array(labels)

X = X / 255.0

y_cat = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test, X_filenames_train, X_filenames_test = train_test_split(
    X, y_cat, image_filenames, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=20, 
                    batch_size=32, 
                    callbacks=[early_stop])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

hog_features = []
for img in X:
    img_uint8 = (img * 255).astype(np.uint8)
    
    gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    
    feature = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(8, 8))
    hog_features.append(feature)

hog_features = np.array(hog_features)

X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(hog_features, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_hog, y_train_hog)

svm_predictions = svm_model.predict(X_test_hog)
svm_accuracy = accuracy_score(y_test_hog, svm_predictions)
print(f"SVM Test Accuracy: {svm_accuracy * 100:.2f}%")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_hog, y_train_hog)

rf_predictions = rf_model.predict(X_test_hog)
rf_accuracy = accuracy_score(y_test_hog, rf_predictions)
print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")

#with open('model.pkl', 'wb') as f:
  #  pickle.dump(model, f)
