import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 28x28 handwritten digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

print(x_train[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())                        # Make the dataset to a flat array
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # "Hidden layers"
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # "Hidden layers"
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', # default?
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy']   
              )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('training_checkpoint',
                                                         save_weights_only=True,
                                                         save_best_only='T,rue',
                                                         monitor='val_loss',
                                                         mode='min'
                                                         )

model.fit(x_train,y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('Num_model')

new_model = tf.keras.models.load_model('Num_model')
predictions = new_model.predict([x_test])

print(np.argmax(predictions[1]))
plt.imshow(x_test[1])
plt.show()
