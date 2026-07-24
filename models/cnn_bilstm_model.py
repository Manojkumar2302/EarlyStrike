import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

class CNNBiLSTM:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # ---- CNN BLOCK (VERY LIGHT) ----
        x = layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.02)
        )(inputs)
        x = layers.SpatialDropout1D(0.5)(x)

        x = layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(0.02)
        )(x)
        x = layers.SpatialDropout1D(0.5)(x)

        # ---- BiLSTM BLOCK (MINIMAL) ----
        x = layers.Bidirectional(
            layers.LSTM(
                32,
                return_sequences=True,
                dropout=0.7,
                recurrent_dropout=0.7
            )
        )(x)

        x = layers.Bidirectional(
            layers.LSTM(
                16,
                dropout=0.7,
                recurrent_dropout=0.7
            )
        )(x)

        # ---- CLASSIFIER ----
        x = layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.02)
        )(x)
        x = layers.Dropout(0.7)(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = models.Model(inputs, outputs)
        return self.model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1
        )
