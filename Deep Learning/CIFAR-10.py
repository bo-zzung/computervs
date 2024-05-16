import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.reshape(50000, 3072)  
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype(np.float32) / 255.0  
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10)

dmlp = Sequential()
dmlp.add(Dense(units=1024, input_shape=(3072,)))
dmlp.add(LeakyReLU(alpha=0.1))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.4))
dmlp.add(Dense(units=512))
dmlp.add(LeakyReLU(alpha=0.1))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.3))
dmlp.add(Dense(units=512))
dmlp.add(LeakyReLU(alpha=0.1))
dmlp.add(BatchNormalization())
dmlp.add(Dropout(0.2))
dmlp.add(Dense(units=10, activation='softmax'))


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])


hist = dmlp.fit(x_train, y_train, batch_size=128, epochs=70, validation_data=(x_test, y_test), verbose=2, callbacks=[lr_scheduler])


accuracy = dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100
print('정확도:', accuracy)


plt.plot(hist.history['accuracy'], 'r--')
plt.plot(hist.history['val_accuracy'], 'b')
plt.title('Accuracy graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.grid(True)
plt.show()


plt.plot(hist.history['loss'], 'r--')
plt.plot(hist.history['val_loss'], 'b')
plt.title('Loss graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.grid(True)
plt.show()
