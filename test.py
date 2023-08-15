# test.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
new_model = tf.keras.models.load_model('Num_model')

# Load and preprocess the data (similar to what you did in main.py)
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Use the pre-trained model for predictions
predictions = new_model.predict([x_test])

# Display prediction for a sample image
sample_index = 0
for i in range(0,len(predictions)):
    sample_index = i
    predicted_label = np.argmax(predictions[sample_index])
    accual_label = y_test[sample_index]
    print("Predicted label:", predicted_label)
    print("Correct answer:", "Yes" if accual_label == predicted_label else "No")

    plt.imshow(x_test[sample_index], cmap='gray')
    plt.pause(2)
    plt.close()
   
