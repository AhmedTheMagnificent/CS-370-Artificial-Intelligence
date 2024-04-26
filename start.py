import tensorflow as tf
import numpy as np

# Generate sample data
numbers = np.arange(100).reshape(100, 1)
labels = np.array([i+2 for i in numbers])

# Define model (consider feature engineering and regularization)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100, input_shape=[1], activation="relu"),
    tf.keras.layers.Dense(units=1, activation="linear")  # Example of L2 regularization
])

# Compile model (consider cost-sensitive learning)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="mse",
              metrics=["accuracy"])  # Consider weighted loss function for class imbalance

# Train model (consider hyperparameter tuning)
model.fit(numbers, labels, epochs=100)

# Evaluate the model
loss, accuracy = model.evaluate(numbers, labels)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(numbers)
predictions = np.round(predictions).flatten()
for num, pred in zip(numbers, predictions):
    print(f'Number: {num}, Output: {pred}')
