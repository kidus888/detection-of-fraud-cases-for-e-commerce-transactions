import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def build_cnn(input_shape):
    """Build a Convolutional Neural Network (CNN) for fraud detection."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_shape):
    """Build a Recurrent Neural Network (RNN) for fraud detection."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    """Build an LSTM model for fraud detection."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    """Train CNN, RNN, or LSTM models."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    
    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)
    
    return accuracy, conf_matrix, report
