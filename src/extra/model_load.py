try:
    import numpy as np
    import os
    import tensorflow as tf
    import json
    from tensorflow.keras import layers, Model, regularizers  # type: ignore
except ImportError:
    raise ImportError("TensorFlow is not installed. Please install TensorFlow to use this module.")


# ============================================
# Step 1: Recreate Model Architecture
# ============================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, d_model, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]  # type: ignore
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        return x + pos_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maxlen": self.maxlen,
                "d_model": self.d_model,
            }
        )
        return config


def attention_block(x, d_model, num_heads, dff, dropout_rate=0.3):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = layers.Dense(dff, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x


def build_cnn_attention_model(input_shape, num_classes, dropout_rate=0.3, l2_reg=1e-4):
    inp = layers.Input(shape=input_shape)

    def conv_block(x, filters, kernel_size):
        x = layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        return x

    b1 = conv_block(inp, 64, 3)
    b2 = conv_block(inp, 64, 5)
    b3 = conv_block(inp, 64, 7)
    x = layers.Concatenate(axis=-1)([b1, b2, b3])

    d_model = 128
    x = layers.Conv1D(
        d_model, kernel_size=1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)

    x = PositionalEmbedding(maxlen=input_shape[0] // 2, d_model=d_model)(x)

    for _ in range(2):
        x = attention_block(x, d_model=d_model, num_heads=4, dff=256, dropout_rate=dropout_rate)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=outputs)
    return model


class CNNAttentionModelLoader:
    """Object-oriented loader for the CNN + Attention model.

    Usage:
        loader = CNNAttentionModelLoader()
        loader.load()  # optional if auto_load=False
        model = loader.model
        preds = loader.predict_batch(X)
    """

    def __init__(self, save_dir: str | None = None, auto_load: bool = True):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)

        self.possible_paths = [
            os.path.join(self.project_root, "saved_models"),
            os.path.join(self.project_root, "extracted_models", "saved_models"),
            "saved_models",
            "extracted_models/saved_models",
        ]

        self.save_dir = save_dir
        self.config = None
        self.input_shape = None
        self.num_classes = None
        self.model = None

        if auto_load:
            self.load()

    def find_save_dir(self) -> str:
        if self.save_dir and os.path.exists(self.save_dir):
            return self.save_dir

        for path in self.possible_paths:
            if os.path.exists(path):
                print(f"Found models directory at: {os.path.abspath(path)}")
                self.save_dir = path
                return path

        raise FileNotFoundError("saved_models directory not found")

    def load_config(self) -> None:
        save_dir = self.find_save_dir()
        config_path = os.path.join(save_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config_str = f.read()

            self.config = json.loads(model_config_str)
            input_shape_list = self.config["config"]["layers"][0]["config"]["batch_input_shape"][1:]
            self.input_shape = tuple(input_shape_list)

            last_layer = self.config["config"]["layers"][-1]
            self.num_classes = last_layer["config"]["units"]

            print(f"Loaded model configuration")
            print(f"Input shape: {self.input_shape}")
            print(f"Number of classes: {self.num_classes}")
        else:
            print("Model config not found. Using default values.")
            print("Please update input_shape and num_classes if these are incorrect!")
            self.input_shape = (768, 1)
            self.num_classes = 5

    def build_model(self) -> None:
        if self.input_shape is None or self.num_classes is None:
            raise RuntimeError("Model configuration is not loaded")

        print("\nRebuilding model architecture...")
        self.model = build_cnn_attention_model(input_shape=self.input_shape, num_classes=self.num_classes)

    def load_weights(self) -> bool:
        save_dir = self.find_save_dir()
        weights_path = os.path.join(save_dir, "cnn_attention_weights.weights.h5")

        if os.path.exists(weights_path):
            print(f"Loading weights from: {weights_path}")
            try:
                assert self.model is not None
                # Call load_weights only if the underlying model supports it
                load_fn = getattr(self.model, "load_weights", None)
                if callable(load_fn):
                    load_fn(weights_path)
                else:
                    raise AttributeError("Underlying model does not support load_weights")
                print("Weights loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load weights: {str(e)}")
                return False
        return False

    class _SavedModelWrapper:
        def __init__(self, infer_func, input_shape, num_classes):
            self.infer = infer_func
            self.input_shape = (None,) + input_shape
            self.output_shape = (None, num_classes)

        def predict(self, x, verbose=0):
            if isinstance(x, np.ndarray):
                x = tf.convert_to_tensor(x, dtype=tf.float32)

            output = self.infer(x)
            result = list(output.values())[0]
            return result.numpy()

        def __call__(self, x):
            return self.predict(x)

    def try_load_savedmodel(self) -> bool:
        save_dir = self.find_save_dir()
        savedmodel_path = os.path.join(save_dir, "cnn_attention_savedmodel")
        if os.path.exists(savedmodel_path):
            try:
                loaded = tf.saved_model.load(savedmodel_path)
                infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]  # type: ignore
                self.model = self._SavedModelWrapper(infer, self.input_shape, self.num_classes)
                print("Model loaded from SavedModel format!")
                return True
            except Exception as e:
                print(f"Failed to load from SavedModel: {str(e)}")
                return False
        return False

    def load(self) -> None:
        # orchestrate loading
        self.load_config()
        self.build_model()

        weights_ok = self.load_weights()
        if not weights_ok:
            print("\nTrying to load from SavedModel format...")
            saved_ok = self.try_load_savedmodel()
            if not saved_ok:
                raise RuntimeError("Failed to load model weights from any source")

        # compile if possible
        compile_fn = getattr(self.model, "compile", None)
        if callable(compile_fn):
            compile_fn(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        print("\n" + "=" * 50)
        print("MODEL LOADED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Input shape: {getattr(self.model, 'input_shape', self.input_shape)}")
        print(f"Output shape: {getattr(self.model, 'output_shape', getattr(self.model, 'output_shape', None))}")

    # --------------------------------------------------
    # Inference helpers
    # --------------------------------------------------
    @staticmethod
    def preprocess_data(data):
        """Preprocess input data. Update this to match training preprocessing."""
        return data

    @staticmethod
    def _ensure_batch_dims(sample, input_shape):
        # Ensure sample has shape (1, *input_shape)
        if isinstance(sample, np.ndarray):
            if sample.ndim == 2:
                sample = np.expand_dims(sample, axis=0)
            elif sample.ndim == 1:
                sample = np.expand_dims(sample, axis=0)
                sample = np.expand_dims(sample, axis=-1)
        return sample

    def predict_single_sample(self, sample):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        sample = self._ensure_batch_dims(sample, self.input_shape)
        probabilities = self.model.predict(sample, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        return predicted_class, confidence, probabilities

    def predict_batch(self, data_batch):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        all_probabilities = self.model.predict(data_batch, verbose=0)
        predictions = np.argmax(all_probabilities, axis=1)
        confidences = np.max(all_probabilities, axis=1)
        return predictions, confidences, all_probabilities


if __name__ == "__main__":
    # Quick demo when run as a script
    loader = CNNAttentionModelLoader()

    # Create dummy data for testing (guard if input_shape wasn't set)
    example_shape = loader.input_shape or (768, 1)
    dummy_sample = np.random.randn(*example_shape)
    dummy_batch = np.random.randn(5, *example_shape)

    print("\nTesting single sample prediction...")
    predicted_class, confidence, probs = loader.predict_single_sample(dummy_sample)
    print(f"✓ Predicted Class: {predicted_class}")
    print(f"✓ Confidence: {confidence:.4f}")
    print(f"✓ All Probabilities: {probs}")

    print("\nTesting batch prediction...")
    predictions, confidences, _ = loader.predict_batch(dummy_batch)
    print(f"✓ Predictions: {predictions}")
    print(f"✓ Confidences: {confidences}")

    print("\nREADY FOR INFERENCE!")