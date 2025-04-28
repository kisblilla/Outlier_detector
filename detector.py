import json
import numpy as np
import tensorflow as tf

# Default configuration constants

SEQ_LEN = 5
BURN_IN = SEQ_LEN
THRESHOLD = 0.06
WEIGHTS_PATH = 'models/autoencoder.weights.h5'
MIN_V = 0.0
MAX_V = 8449.3107

class TumorAutoencoder(tf.keras.Model):
    def __init__(self, sequence_length, num_features=1):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=-10, input_shape=(None, num_features)),
            tf.keras.layers.LSTM(128, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(32, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.RepeatVector(sequence_length),
            tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_features))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class OutlierDetector:
    """
    A real-time outlier detector for tumor volume time series using a pretrained autoencoder.

    Default configuration matches test.py constants.
    """
    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
        stats: dict = {'min': MIN_V, 'max': MAX_V},
        sequence_length: int = SEQ_LEN,
        threshold: float = THRESHOLD,
        burn_in: int = BURN_IN
    ):
        # Store parameters
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.burn_in = burn_in
        self.min_v = float(stats['min'])
        self.max_v = float(stats['max'])

        # Build and load model
        self.model = TumorAutoencoder(sequence_length, num_features=1)
        # Dummy call to build layers
        self.model(tf.zeros((1, sequence_length, 1)))
        self.model.load_weights(weights_path)

    def detect(self, volumes_json: str) -> str:
        """
        Detect outliers in a JSON-encoded list of tumor volumes.

        Args:
            volumes_json: JSON string of a list of floats (tumor volumes).

        Returns:
            JSON string of a list of booleans, same length as input, where True indicates an outlier.
        """
        # Parse input volumes
        volumes = json.loads(volumes_json)

        vals = np.array(volumes, dtype=float)

        # Normalize tumor volumes
        norm = (vals - self.min_v) / (self.max_v - self.min_v)

        flags = []
        window = []
        seq_len = self.sequence_length

        for i, v in enumerate(norm):
            window.append(v)

            if i < self.burn_in or len(window) < seq_len:
                flags.append(False)
                continue

            # Take the last 'sequence_length' points
            seq = window[-seq_len:]
            inp = np.array(seq).reshape(1, seq_len, 1)

            # Predict and compute error on the last point
            recon = self.model.predict(inp, verbose=0)
            pred_last = recon[0, -1, 0]
            err = abs(seq[-1] - pred_last)
            flags.append(bool(err > self.threshold))

        return json.dumps(flags)
