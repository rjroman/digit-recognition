
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""# Loading Data"""

train_data = pd.read_csv("K:/syncintern/digit recognition/digit-recognizer/train.csv")
train_data.shape

test_data = pd.read_csv("K:/syncintern/digit recognition/digit-recognizer/test.csv")
test_data.shape

train_data.head()

test_data.head()

train_data.isnull().sum().sum()

plt.figure()
sns.countplot(x=train_data.label)

"""# Reshape the data
reshape to 28x28 to apply CNN
"""

# drop target and then reshape remaining data
train_data_2d=train_data.drop('label', axis=1)
train_data_2d=train_data_2d.values.reshape(-1,28,28,1)

test_data_2d=test_data.values.reshape(-1,28,28,1)

for i in range(5):
    plt.figure(figsize=(1,1))
    plt.imshow(train_data_2d[i])
    plt.show()

train_data.label[:5]

"""# Splitting Data"""

X = train_data_2d
y = train_data.label

X_test = test_data_2d

# Normalize the data
X = X/255.0
X_test = X_test/255.0

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

train_shape = X_train.shape
train_shape

y_train.shape

"""# Modeling"""

model = keras.Sequential([
#     CNN base
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
#     dense head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
#     callbacks=[early_stopping],
    epochs=15,
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[:,['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(),
              history_df['val_accuracy'].max()))

"""# Confusion Matrix"""

pred = model.predict(X_valid)
pred_valid = pd.DataFrame([np.argmax(i) for i in pred])
cm = tf.math.confusion_matrix(labels=y_valid,predictions=pred_valid)
cm

plt.figure()
sns.heatmap(cm, annot=True,fmt='d')

"""# Predictions
predictions on test data
"""

preds = model.predict(X_test)
predictions = [np.argmax(i) for i in preds]
predictions = pd.Series(predictions,name="Label")

output = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

output.to_csv("submission.csv",index=False)