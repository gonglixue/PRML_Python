import tensorflow as tf
from tensorflow import keras
import numpy as np

class FlowerClassifier():
    def __init__(self):
        self.keras_model = self.build()

    def build(self):

        inputs = keras.layers.Input(shape=(4, ))
        x = keras.layers.Dense(10, activation='relu', name='dense_1')(inputs)
        x = keras.layers.Dense(20, activation='relu', name='dense_2')(x)
        x = keras.layers.Dense(10, activation='relu', name='dense_3')(x)

        predictions = keras.layers.Dense(3, activation='softmax', name='output')(x)

        model = keras.models.Model(inputs=inputs, outputs=predictions)
        return model

    def train(self, learning_rate, momentum, batch_size, epochs, train_set_csv_path, val_set_csv_path):
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=train_set_csv_path,
            target_dtype=np.int,
            features_dtype=np.float32
        )
        val_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=val_set_csv_path,
            target_dtype=np.int,
            features_dtype=np.float32
        )

        train_data = training_set.data
        train_target = training_set.target
        test_data = val_set.data
        test_target = val_set.target

        train_target = keras.utils.to_categorical(train_target, num_classes=3)
        test_target = keras.utils.to_categorical(test_target, num_classes=3)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir="iris_log", histogram_freq=0, write_grads=True, write_images=False),
            keras.callbacks.ModelCheckpoint("./iris_ckpt", save_weights_only=True, verbose=0)
        ]

        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
        self.keras_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.keras_model.fit(train_data, train_target, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        score = self.keras_model.evaluate(test_data, test_target, batch_size=batch_size)
        test_predict = self.keras_model.predict(test_data)
        print(np.argmax(test_predict, axis=1))

def test():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename='../../iris_training.csv',
        target_dtype=np.int,
        features_dtype=np.float32
    )
    val_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename='../../iris_test.csv',
        target_dtype=np.int,
        features_dtype=np.float32
    )

if __name__ == '__main__':
    Classifier = FlowerClassifier()
    Classifier.train(learning_rate=0.005, momentum=0.9, batch_size=32, epochs=50,
                     train_set_csv_path='../../iris_training.csv', val_set_csv_path='../../iris_test.csv')