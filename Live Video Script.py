import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import ops
import time
from scipy import stats

# Dummy class weights (equal, since inference doesn't use them)
class_weights_tensor = tf.constant([1.0] * 8, dtype=tf.float32)

import os
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import ops
from tensorflow.keras import regularizers
import numpy as np

# Define all custom losses and metrics (must match the saving code, with registrations)
@tf.keras.utils.register_keras_serializable()
class MaskedFocalSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.05, num_classes=8, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.class_weights = tf.constant(np.ones(num_classes, dtype=np.float32))

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, -1)
        mask_f = tf.cast(mask, tf.float32)
        y_true_safe = tf.where(mask, y_true, 0)
        alpha = tf.gather(self.class_weights, y_true_safe)
        y_true_one_hot = tf.one_hot(y_true_safe, depth=self.num_classes)
        y_true_one_hot = (1 - self.label_smoothing) * y_true_one_hot + (self.label_smoothing / self.num_classes)
        ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred, from_logits=False)
        pt = tf.exp(-ce)
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce
        focal_loss = tf.where(mask, focal_loss, tf.zeros_like(focal_loss))
        return tf.reduce_sum(focal_loss) / (tf.reduce_sum(mask_f) + tf.keras.backend.epsilon())

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
            'num_classes': self.num_classes
        })
        return config

@tf.keras.utils.register_keras_serializable()
class VALoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask_v = tf.greater(y_true[:, 0], -1.5)
        mask_a = tf.greater(y_true[:, 1], -1.5)
        v_true = tf.boolean_mask(y_true[:, 0], mask_v)
        v_pred = tf.boolean_mask(y_pred[:, 0], mask_v)
        a_true = tf.boolean_mask(y_true[:, 1], mask_a)
        a_pred = tf.boolean_mask(y_pred[:, 1], mask_a)
        ccc_v = self.ccc(v_true, v_pred)
        ccc_a = self.ccc(a_true, a_pred)
        return (1 - ccc_v + 1 - ccc_a) / 2

    def ccc(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mu_true = tf.reduce_mean(y_true)
        mu_pred = tf.reduce_mean(y_pred)
        var_true = tf.math.reduce_variance(y_true)
        var_pred = tf.math.reduce_variance(y_pred)
        cov = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred))
        return (2 * cov) / (var_true + var_pred + (mu_true - mu_pred)**2 + 1e-6)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class AULoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, -1)
        y_true_masked = tf.where(mask, tf.cast(y_true, tf.float32), 0.0)
        return tf.keras.losses.binary_crossentropy(y_true_masked, y_pred)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class MulticlassF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name='f1', **kwargs):
        super(MulticlassF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.per_class_tp = [self.add_weight(name=f'tp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.per_class_fp = [self.add_weight(name=f'fp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.per_class_fn = [self.add_weight(name=f'fn_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        for i in range(self.num_classes):
            class_mask_true = tf.equal(y_true, i)
            class_mask_pred = tf.equal(y_pred_classes, i)
            tp = tf.reduce_sum(tf.cast(tf.logical_and(class_mask_true, class_mask_pred), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(class_mask_true), class_mask_pred), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(class_mask_true, tf.logical_not(class_mask_pred)), tf.float32))
            self.per_class_tp[i].assign_add(tp)
            self.per_class_fp[i].assign_add(fp)
            self.per_class_fn[i].assign_add(fn)

    def result(self):
        prec = [self.per_class_tp[i] / (self.per_class_tp[i] + self.per_class_fp[i] + tf.keras.backend.epsilon()) for i in range(self.num_classes)]
        rec = [self.per_class_tp[i] / (self.per_class_tp[i] + self.per_class_fn[i] + tf.keras.backend.epsilon()) for i in range(self.num_classes)]
        f1_per_class = [2 * (prec[i] * rec[i]) / (prec[i] + rec[i] + tf.keras.backend.epsilon()) for i in range(self.num_classes)]
        return tf.reduce_mean(tf.stack(f1_per_class))

    def reset_state(self):
        for i in range(self.num_classes):
            self.per_class_tp[i].assign(0.0)
            self.per_class_fp[i].assign(0.0)
            self.per_class_fn[i].assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

@tf.keras.utils.register_keras_serializable()
class MulticlassPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name='precision', **kwargs):
        super(MulticlassPrecision, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.per_class_tp = [self.add_weight(name=f'tp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.per_class_fp = [self.add_weight(name=f'fp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, -1)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        y_pred_classes = tf.argmax(y_pred_masked, axis=-1, output_type=tf.int32)
        for i in range(self.num_classes):
            class_mask_true = tf.equal(y_true_masked, i)
            class_mask_pred = tf.equal(y_pred_classes, i)
            tp = tf.reduce_sum(tf.cast(tf.logical_and(class_mask_true, class_mask_pred), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(class_mask_true), class_mask_pred), tf.float32))
            self.per_class_tp[i].assign_add(tp)
            self.per_class_fp[i].assign_add(fp)

    def result(self):
        precision_per_class = [self.per_class_tp[i] / (self.per_class_tp[i] + self.per_class_fp[i] + tf.keras.backend.epsilon()) for i in range(self.num_classes)]
        return tf.reduce_mean(tf.stack(precision_per_class))

    def reset_state(self):
        for i in range(self.num_classes):
            self.per_class_tp[i].assign(0.0)
            self.per_class_fp[i].assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

@tf.keras.utils.register_keras_serializable()
class MulticlassRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name='recall', **kwargs):
        super(MulticlassRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.per_class_tp = [self.add_weight(name=f'tp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.per_class_fn = [self.add_weight(name=f'fn_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, -1)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        y_pred_classes = tf.argmax(y_pred_masked, axis=-1, output_type=tf.int32)
        for i in range(self.num_classes):
            class_mask_true = tf.equal(y_true_masked, i)
            class_mask_pred = tf.equal(y_pred_classes, i)
            tp = tf.reduce_sum(tf.cast(tf.logical_and(class_mask_true, class_mask_pred), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(class_mask_true, tf.logical_not(class_mask_pred)), tf.float32))
            self.per_class_tp[i].assign_add(tp)
            self.per_class_fn[i].assign_add(fn)

    def result(self):
        recall_per_class = [self.per_class_tp[i] / (self.per_class_tp[i] + self.per_class_fn[i] + tf.keras.backend.epsilon()) for i in range(self.num_classes)]
        return tf.reduce_mean(tf.stack(recall_per_class))

    def reset_state(self):
        for i in range(self.num_classes):
            self.per_class_tp[i].assign(0.0)
            self.per_class_fn[i].assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

@tf.keras.utils.register_keras_serializable()
class VA_CCC(tf.keras.metrics.Metric):
    def __init__(self, name='va_ccc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ccc_v_sum = self.add_weight(shape=(), name='ccc_v_sum', initializer='zeros')
        self.ccc_a_sum = self.add_weight(shape=(), name='ccc_a_sum', initializer='zeros')
        self.count = self.add_weight(shape=(), name='count', initializer='zeros')

    def update_state(self, y_true_va, y_pred_va, sample_weight=None):
        mask_v = tf.greater(y_true_va[:, 0], -1.5)
        mask_a = tf.greater(y_true_va[:, 1], -1.5)
        v_true = tf.boolean_mask(y_true_va[:, 0], mask_v)
        v_pred = tf.boolean_mask(y_pred_va[:, 0], mask_v)
        a_true = tf.boolean_mask(y_true_va[:, 1], mask_a)
        a_pred = tf.boolean_mask(y_pred_va[:, 1], mask_a)
        ccc_v = self.ccc(v_true, v_pred)
        ccc_a = self.ccc(a_true, a_pred)
        self.ccc_v_sum.assign_add(ccc_v)
        self.ccc_a_sum.assign_add(ccc_a)
        self.count.assign_add(1.0)

    def ccc(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mu_true = tf.reduce_mean(y_true)
        mu_pred = tf.reduce_mean(y_pred)
        var_true = tf.math.reduce_variance(y_true)
        var_pred = tf.math.reduce_variance(y_pred)
        cov = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred))
        return (2 * cov) / (var_true + var_pred + (mu_true - mu_pred)**2 + 1e-6)

    def result(self):
        return (self.ccc_v_sum + self.ccc_a_sum) / (2 * self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.ccc_v_sum.assign(0.0)
        self.ccc_a_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class AU_F1(tf.keras.metrics.Metric):
    def __init__(self, name='au_f1', num_au=12, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_au = num_au
        self.per_au_tp = [self.add_weight(name=f'tp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_au)]
        self.per_au_fp = [self.add_weight(name=f'fp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_au)]
        self.per_au_fn = [self.add_weight(name=f'fn_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_au)]

    def update_state(self, y_true_au, y_pred_au, sample_weight=None):
        mask = tf.not_equal(y_true_au, -1)
        y_true_masked = tf.where(mask, tf.cast(y_true_au, tf.float32), tf.zeros_like(y_true_au, dtype=tf.float32))
        y_pred_bin = tf.round(y_pred_au)
        for i in range(self.num_au):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:, i], tf.int32) == 1, tf.cast(y_pred_bin[:, i], tf.int32) == 1), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:, i], tf.int32) == 0, tf.cast(y_pred_bin[:, i], tf.int32) == 1), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:, i], tf.int32) == 1, tf.cast(y_pred_bin[:, i], tf.int32) == 0), tf.float32))
            self.per_au_tp[i].assign_add(tp)
            self.per_au_fp[i].assign_add(fp)
            self.per_au_fn[i].assign_add(fn)

    def result(self):
        prec = [self.per_au_tp[i] / (self.per_au_tp[i] + self.per_au_fp[i] + tf.keras.backend.epsilon()) for i in range(self.num_au)]
        rec = [self.per_au_tp[i] / (self.per_au_tp[i] + self.per_au_fn[i] + tf.keras.backend.epsilon()) for i in range(self.num_au)]
        f1_per_au = [2 * (prec[i] * rec[i]) / (prec[i] + rec[i] + tf.keras.backend.epsilon()) for i in range(self.num_au)]
        return tf.reduce_mean(tf.stack(f1_per_au))

    def reset_state(self):
        for i in range(self.num_au):
            self.per_au_tp[i].assign(0.0)
            self.per_au_fp[i].assign(0.0)
            self.per_au_fn[i].assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({'num_au': self.num_au})
        return config

# Now load the model
models_dir = 'final_model'
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"Models directory '{models_dir}' not found. Ensure .keras files are placed there.")

keras_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
if not keras_files:
    raise FileNotFoundError(f"No .keras files found in '{models_dir}'. Expected files like full_mtl_model.keras.")

model_path = os.path.join(models_dir, keras_files[0])  # Assuming the first (or only) .keras file

try:
    # Since classes are registered via decorators, no custom_objects needed; but include for safety
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'MaskedFocalSparseCategoricalCrossentropy': MaskedFocalSparseCategoricalCrossentropy,
            'VALoss': VALoss,
            'AULoss': AULoss,
            'MulticlassF1Score': MulticlassF1Score,
            'MulticlassPrecision': MulticlassPrecision,
            'MulticlassRecall': MulticlassRecall,
            'VA_CCC': VA_CCC,
            'AU_F1': AU_F1
        }
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model.")

# Emotion mapping (including "Other")
EMOTION_MAP = {0: "Neutral", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise", 7: "Other"}

def translate_outputs(expr_pred, va_pred, au_pred):
    emotions = ["Neutral", "Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Other"]
    
    # Expr probs: include all 8 classes (balanced, no exclusion of "Other")
    total_sum = np.sum(expr_pred)
    if total_sum > 0:
        expr_probs = expr_pred / total_sum
    else:
        expr_probs = np.ones(8) / 8  # Fallback uniform
    
    # AU active set: threshold at 0.5 for more activations
    O = set(np.where(au_pred > 0.5)[0])
    
    # AU prototypes updated based on standard FACS associations for available AUs
    au_dict = {
        "Neutral": {},
        "Anger": {2: 1.0, 4: 1.0, 8: 1.0},  # AU4, AU7, AU23
        "Disgust": {5: 1.0, 7: 1.0},  # AU10, AU15
        "Fear": {0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0, 10: 0.5, 11: 1.0},  # AU1, AU2, AU4, AU7, AU25, AU26
        "Happy": {3: 1.0, 6: 1.0},  # AU6, AU12
        "Sad": {0: 1.0, 2: 1.0, 7: 1.0},  # AU1, AU4, AU15
        "Surprise": {0: 1.0, 1: 1.0, 10: 0.5, 11: 1.0},  # AU1, AU2, AU25, AU26
        "Other": {}  # Empty for misc/contempt-like
    }
    
    au_scores = {}
    for em in emotions:
        aus = au_dict.get(em, {})
        if not aus:
            score = -len(O) * 0.5  # Milder penalty for neutral/other
            total_w = max(len(O), 1)
        else:
            TP = sum(aus.get(au, 0) for au in O)
            FP = len(O - set(aus.keys())) * 0.5  # Reduced penalty for extra AUs
            FN = sum(aus.values()) - sum(aus.get(au, 0) for au in O)
            score = TP - FP - FN
            total_w = sum(aus.values()) + len(aus)
        au_scores[em] = score / total_w if total_w else score
    
    # AU probs
    values = np.array(list(au_scores.values()))
    if np.all(values == 0):
        au_probs = np.ones(len(values)) / len(values)
    else:
        exp_v = np.exp(values - np.max(values))
        au_probs = exp_v / exp_v.sum()
    
    # VA prototypes updated to more accurate averages from research
    va_points = {
        "Neutral": (0.0, 0.0),
        "Anger": (-0.51, 0.59),
        "Disgust": (-0.60, 0.35),
        "Fear": (-0.64, 0.60),
        "Happy": (0.81, 0.51),
        "Sad": (-0.63, -0.27),
        "Surprise": (0.40, 0.67),
        "Other": (0.0, 0.0)
    }
    
    # Optimal VA ranges for each emotion
    va_ranges = {
        "Neutral": {'v': [-0.2, 0.2], 'a': [-0.2, 0.2]},
        "Anger": {'v': [-1.0, -0.3], 'a': [0.3, 1.0]},
        "Disgust": {'v': [-1.0, -0.3], 'a': [-0.2, 0.6]},
        "Fear": {'v': [-1.0, -0.3], 'a': [0.3, 1.0]},
        "Happy": {'v': [0.3, 1.0], 'a': [0.2, 1.0]},
        "Sad": {'v': [-1.0, -0.3], 'a': [-1.0, 0.0]},
        "Surprise": {'v': [0.0, 0.6], 'a': [0.3, 1.0]},
        "Other": {'v': [-0.5, 0.5], 'a': [-0.5, 0.5]}
    }
    
    dists = [np.sqrt((va_pred[0] - va_points[em][0])**2 + (va_pred[1] - va_points[em][1])**2) for em in emotions]
    va_scores = -np.array(dists)
    
    # Boost VA scores if within optimal range
    for i, em in enumerate(emotions):
        r = va_ranges[em]
        if r['v'][0] <= va_pred[0] <= r['v'][1] and r['a'][0] <= va_pred[1] <= r['a'][1]:
            va_scores[i] += 1.0  # Boost for being in range
    
    exp_v = np.exp(va_scores - np.max(va_scores))
    va_probs = exp_v / exp_v.sum()
    
    # Fusion weights (slightly increased for expr to leverage direct prediction)
    w = [0.4, 0.4, 0.2]  # expr, au, va
    final_scores = w[0] * expr_probs + w[1] * au_probs + w[2] * va_probs
    
    # Add bias to prioritize Happy and Neutral
    bias = np.array([0.15, 0.05, 0.0, -0.03, 0.14, -0.15, 0.0, 0.1])  # Boost for Neutral (0) and Happy (4)
    final_scores += bias
    
    # Entropy check for adaptive adjustment
    temp_scores = final_scores  # Pre-temp for entropy
    exp_temp = np.exp(temp_scores - np.max(temp_scores))
    temp_probs = exp_temp / exp_temp.sum()
    entropy = stats.entropy(temp_probs)
    if entropy > 1.5:  # High uncertainty: boost AU/VA
        w = [0.3, 0.35, 0.35]
        final_scores = w[0] * expr_probs + w[1] * au_probs + w[2] * va_probs + bias
        T = 1.0  # Soften
    else:
        T = 0.8  # Sharpen
    
    # Apply temperature
    final_scores /= T
    exp_f = np.exp(final_scores - np.max(final_scores))
    final_probs = exp_f / exp_f.sum()
    
    # Label via argmax
    label = emotions[np.argmax(final_probs)]
    
    # Mood modifiers (unchanged)
    valence, arousal = va_pred
    active_aus_count = len(O)
    mood = ""
    if valence > 0.3:
        mood = "Positive "
    elif valence < -0.3:
        mood = "Negative "
    if arousal > 0.4 and active_aus_count > 3:
        mood += "Intense "
    
    return mood + label

# Load YOLOv11 face model
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
yolo_model = YOLO(model_path)

# Webcam processing
cap = cv2.VideoCapture(0)  # 0 for default webcam
fps_start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLO detection
    results = yolo_model.track(frame, persist=True, tracker="botsort.yaml")

    annotated_frame = frame.copy()
    faces = []
    boxes = []
    for r in results:
        for box in r.boxes:
            if box.cls == 0:  # Face
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (224, 224))
                    crop_preprocessed = preprocess_input(crop_resized.astype(np.float32))
                    faces.append(crop_preprocessed)

    if faces:
        faces_batch = np.array(faces)
        
        # Ensemble predictions: average across models
        all_expr = []
        all_va = []
        all_au = []
        try:
            preds = model.predict(faces_batch, verbose=0)
            all_expr.append(preds['expr'])
            all_va.append(preds['va'])
            all_au.append(preds['au'])
        except Exception as e:
            print(f"Error predicting with model on frame: {e}")
            continue  # Skip this model for this frame
        
        if all_expr:  # Only if at least one model succeeded
            expr_preds = np.mean(np.array(all_expr), axis=0)
            va_preds = np.mean(np.array(all_va), axis=0)
            au_preds = np.mean(np.array(all_au), axis=0)
        else:
            continue  # Skip frame if no predictions

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            label = translate_outputs(expr_preds[i], va_preds[i], au_preds[i])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display
    cv2.imshow('Emotion Detection', annotated_frame)

    frame_count += 1
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - fps_start)
        print(f"Average FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()