import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import gdown
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import ops
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import random
import tempfile
from google.cloud import storage
from matplotlib.widgets import Button
import time

# run this: gcloud auth application-default login

def authenticate_implicit_with_adc(project_id="facial-expression-466800"):
    storage_client = storage.Client(project=project_id)
    buckets = storage_client.list_buckets()
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
    print("Listed all storage buckets.")

authenticate_implicit_with_adc()
    
# Define AU labels (assuming standard Action Units mapping; adjust if needed)
AU_LABELS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU17', 'AU25', 'AU26']

# Dummy class weights (equal, since inference doesn't use them)
class_weights_tensor = tf.constant([1.0] * 8, dtype=tf.float32)

# Custom losses/metrics (required for compile/load)
@tf.keras.utils.register_keras_serializable()
def masked_focal_sparse_categorical_crossentropy(gamma=3.0, label_smoothing=0.05):
    def loss_fn(y_true_expr, y_pred_expr):
        y_true_expr = tf.cast(y_true_expr, tf.int32)
        mask = tf.not_equal(y_true_expr, -1)
        mask_f = tf.cast(mask, tf.float32)
        y_true_safe = tf.where(mask, y_true_expr, 0)
        alpha = tf.gather(class_weights_tensor, y_true_safe)
        y_true_one_hot = tf.one_hot(y_true_safe, depth=8)
        y_true_one_hot = (1 - label_smoothing) * y_true_one_hot + (label_smoothing / 8)
        ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred_expr, from_logits=False)
        pt = tf.exp(-ce)
        focal_loss = alpha * ((1 - pt) ** gamma) * ce
        focal_loss = tf.where(mask, focal_loss, tf.zeros_like(focal_loss))
        return tf.reduce_sum(focal_loss) / (tf.reduce_sum(mask_f) + tf.keras.backend.epsilon())
    return loss_fn

def ccc(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mu_true = tf.reduce_mean(y_true)
    mu_pred = tf.reduce_mean(y_pred)
    var_true = tf.math.reduce_variance(y_true)
    var_pred = tf.math.reduce_variance(y_pred)
    cov = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred))
    return (2 * cov) / (var_true + var_pred + (mu_true - mu_pred)**2 + 1e-6)

def va_loss(y_true_va, y_pred_va):
    mask_v = tf.greater(y_true_va[:,0], -1.5)
    mask_a = tf.greater(y_true_va[:,1], -1.5)
    v_true = tf.boolean_mask(y_true_va[:,0], mask_v)
    v_pred = tf.boolean_mask(y_pred_va[:,0], mask_v)
    a_true = tf.boolean_mask(y_true_va[:,1], mask_a)
    a_pred = tf.boolean_mask(y_pred_va[:,1], mask_a)
    ccc_v = ccc(v_true, v_pred)
    ccc_a = ccc(a_true, a_pred)
    return (1 - ccc_v + 1 - ccc_a) / 2

def au_loss(y_true_au, y_pred_au):
    mask = tf.not_equal(y_true_au, -1)
    y_true_masked = tf.where(mask, tf.cast(y_true_au, tf.float32), 0.0)
    return tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_au)

@tf.keras.utils.register_keras_serializable()
class MulticlassF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', **kwargs):
        super(MulticlassF1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true_one_hot = tf.one_hot(y_true, depth=8)
        self.precision.update_state(y_true_one_hot, y_pred, sample_weight)
        self.recall.update_state(y_true_one_hot, y_pred, sample_weight)
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

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

@tf.keras.utils.register_keras_serializable()
class VA_CCC(tf.keras.metrics.Metric):
    def __init__(self, name='va_ccc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ccc_v_sum = self.add_weight(shape=(), name='ccc_v_sum', initializer='zeros')
        self.ccc_a_sum = self.add_weight(shape=(), name='ccc_a_sum', initializer='zeros')
        self.count = self.add_weight(shape=(), name='count', initializer='zeros')

    def update_state(self, y_true_va, y_pred_va, sample_weight=None):
        mask_v = tf.greater(y_true_va[:,0], -1.5)
        mask_a = tf.greater(y_true_va[:,1], -1.5)
        v_true = tf.boolean_mask(y_true_va[:,0], mask_v)
        v_pred = tf.boolean_mask(y_pred_va[:,0], mask_v)
        a_true = tf.boolean_mask(y_true_va[:,1], mask_a)
        a_pred = tf.boolean_mask(y_pred_va[:,1], mask_a)
        self.ccc_v_sum.assign_add(ccc(v_true, v_pred))
        self.ccc_a_sum.assign_add(ccc(a_true, a_pred))
        self.count.assign_add(1.0)

    def result(self):
        return (self.ccc_v_sum + self.ccc_a_sum) / (2 * self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.ccc_v_sum.assign(0.0)
        self.ccc_a_sum.assign(0.0)
        self.count.assign(0.0)

@tf.keras.utils.register_keras_serializable()
class AU_F1(tf.keras.metrics.Metric):
    def __init__(self, name='au_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.per_au_tp = [self.add_weight(name=f'tp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(12)]
        self.per_au_fp = [self.add_weight(name=f'fp_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(12)]
        self.per_au_fn = [self.add_weight(name=f'fn_{i}', shape=(), initializer='zeros', dtype=tf.float32) for i in range(12)]

    def update_state(self, y_true_au, y_pred_au, sample_weight=None):
        mask = tf.not_equal(y_true_au, -1)
        y_true_masked = tf.where(mask, tf.cast(y_true_au, tf.float32), tf.zeros_like(y_true_au, dtype=tf.float32))
        y_pred_bin = tf.round(y_pred_au)

        for i in range(12):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:,i], tf.int32) == 1, tf.cast(y_pred_bin[:,i], tf.int32) == 1), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:,i], tf.int32) == 0, tf.cast(y_pred_bin[:,i], tf.int32) == 1), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true_masked[:,i], tf.int32) == 1, tf.cast(y_pred_bin[:,i], tf.int32) == 0), tf.float32))

            self.per_au_tp[i].assign_add(tp)
            self.per_au_fp[i].assign_add(fp)
            self.per_au_fn[i].assign_add(fn)

    def result(self):
        prec = [self.per_au_tp[i] / (self.per_au_tp[i] + self.per_au_fp[i] + tf.keras.backend.epsilon()) for i in range(12)]
        rec = [self.per_au_tp[i] / (self.per_au_tp[i] + self.per_au_fn[i] + tf.keras.backend.epsilon()) for i in range(12)]
        f1_per_au = [2 * (prec[i] * rec[i]) / (prec[i] + rec[i] + tf.keras.backend.epsilon()) for i in range(12)]
        return tf.reduce_mean(tf.stack(f1_per_au))

    def reset_state(self):
        for i in range(12):
            self.per_au_tp[i].assign(0.0)
            self.per_au_fp[i].assign(0.0)
            self.per_au_fn[i].assign(0.0)

# Emotion mapping
EMOTION_MAP = {0: "Neutral", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise", 7: "Other"}
def translate_outputs(expr_pred, va_pred, au_pred):
    emotions = ["Neutral", "Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    
    # Expression probabilities (exclude "Other" and normalize)
    expr_probs = expr_pred[:7] / np.sum(expr_pred[:7])
    
    # AU active set
    O = set(np.where(au_pred > 0.5)[0])
    
    # AU prototypes
    au_dict = {
        "Neutral": {},
        "Anger": {2: 1.0, 4: 1.0, 9: 1.0, 10: 0.5},
        "Disgust": {5: 1.0, 7: 1.0, 8: 1.0},
        "Fear": {0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0, 11: 0.5},
        "Happy": {3: 1.0, 6: 1.0},
        "Sad": {0: 1.0, 2: 1.0, 7: 1.0, 8: 0.5},
        "Surprise": {0: 1.0, 1: 1.0, 11: 1.0},
    }
    
    au_scores = {}
    for em in emotions:
        aus = au_dict.get(em, {})
        if not aus:
            score = -len(O)
            total_w = max(len(O), 1)
        else:
            TP = sum(aus.get(au, 0) for au in O)
            FP = len(O - set(aus.keys()))
            FN = sum(aus.values()) - sum(aus.get(au, 0) for au in O)
            score = TP - FP - FN
            total_w = sum(aus.values()) + len(aus)
        au_scores[em] = score / total_w if total_w else score
    
    # AU probabilities
    values = np.array(list(au_scores.values()))
    if np.all(values == 0):
        au_probs = np.ones(len(values)) / len(values)
    else:
        exp_v = np.exp(values - np.max(values))
        au_probs = exp_v / exp_v.sum()
    
    # VA prototypes
    va_points = {
        "Neutral": (0, 0),
        "Anger": (-0.8, 0.8),
        "Disgust": (-0.6, 0.4),
        "Fear": (-0.6, 0.8),
        "Happy": (0.8, 0.5),
        "Sad": (-0.8, -0.5),
        "Surprise": (0.4, 0.8),
    }
    
    dists = [np.sqrt((va_pred[0] - va_points[em][0])**2 + (va_pred[1] - va_points[em][1])**2) for em in emotions]
    va_scores = -np.array(dists)
    exp_v = np.exp(va_scores - np.max(va_scores))
    va_probs = exp_v / exp_v.sum()
    
    # Fusion weights
    w = [0.35, 0.35, 0.3]
    final_scores = w[0] * expr_probs + w[1] * au_probs + w[2] * va_probs
    
    # Entropy check for adaptive adjustment
    temp_scores = final_scores
    exp_temp = np.exp(temp_scores - np.max(temp_scores))
    temp_probs = exp_temp / exp_temp.sum()
    entropy = scipy.stats.entropy(temp_probs)
    if entropy > 1.5:
        w = [0.3, 0.4, 0.3]
        final_scores = w[0] * expr_probs + w[1] * au_probs + w[2] * va_probs
        T = 1.0
    else:
        T = 0.8
    
    # Apply temperature
    final_scores /= T
    exp_f = np.exp(final_scores - np.max(final_scores))
    final_probs = exp_f / exp_f.sum()
    
    # Final label
    label = emotions[np.argmax(final_probs)]
    
    # Mood modifiers
    valence, arousal = va_pred
    active_aus_count = len(O)
    mood = ""
    if valence > 0.3:
        mood = "Positive "
    elif valence < -0.3:
        mood = "Negative "
    if arousal > 0.4 and active_aus_count > 3:
        mood += "Intense "
    
    return mood + label, final_probs, O, emotions, expr_probs, va_pred

# Load YOLOv11 face model
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
yolo_model = YOLO(model_path)

# Download weights if not local
weights_url = "https://drive.google.com/uc?id=1XNfsWhak4whY9smoPU7QnCtECZOlbcw8"
weights_path = "mtl_model_epoch_60.weights.h5"
if not os.path.exists(weights_path):
    print(f"Downloading model weights from {weights_url}")
    try:
        gdown.download(weights_url, weights_path, quiet=False)
    except Exception as e:
        print(f"Error downloading model weights: {e}")
        exit(1)

# Recreate MTL model
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    full_input = Input(shape=(224, 224, 3), name='full')
    base_model = tf.keras.applications.EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_tensor=full_input,
        input_shape=(224, 224, 3),
        name='efficientnetv2-s'
    )
    base_model.trainable = True

    base_output = GlobalAveragePooling2D()(base_model.output)
    fused = Dropout(0.7)(base_output)
    fused_with_seq = ops.expand_dims(fused, axis=1)
    attention = MultiHeadAttention(num_heads=4, key_dim=128)(fused_with_seq, fused_with_seq)
    attention = ops.squeeze(attention, axis=1)
    x = BatchNormalization()(attention)
    x = Dropout(0.7)(x)

    expr_out = Dense(8, activation='softmax', dtype='float32', name='expr')(x)
    va_out = Dense(2, activation='tanh', name='va')(x)
    au_out = Dense(12, activation='sigmoid', name='au')(x)

    mtl_model = Model(inputs=full_input, outputs={'expr': expr_out, 'va': va_out, 'au': au_out})

    # Compile with customs
    mtl_model.compile(
        optimizer='adam',
        loss={'expr': masked_focal_sparse_categorical_crossentropy(), 'va': va_loss, 'au': au_loss},
        metrics={'expr': [tf.keras.metrics.SparseCategoricalAccuracy()], 'va': [VA_CCC()], 'au': [AU_F1()]}
    )

    # Load weights
    try:
        mtl_model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

# GCS setup - replace with your bucket name
BUCKET_NAME = 'facial-expressions_bucket'  # Replace with your actual GCS bucket name

# Download pickled data from Google Drive
pickle_url = "https://drive.google.com/uc?id=1azklojBKds7jJJkGRAD-WxMqstro8ydO"  # Replace with your actual pickle file ID
pickle_path = "pairs_cache.pkl"
if not os.path.exists(pickle_path):
    print(f"Downloading pickle file from {pickle_url}")
    try:
        gdown.download(pickle_url, pickle_path, quiet=False)
    except Exception as e:
        print(f"Error downloading pickle file: {e}")
        exit(1)

# Load pickled data
try:
    with open(pickle_path, 'rb') as f:
        train_pairs, val_pairs = pickle.load(f)
    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs from pickle file")
except Exception as e:
    print(f"Error loading pickle file: {e}")
    if os.path.exists(pickle_path):
        os.unlink(pickle_path)
    exit(1)

# Combine train and validation pairs
data_pairs = train_pairs + val_pairs
NUM_IMAGES = 10
if len(data_pairs) < NUM_IMAGES:
    raise ValueError(f"Not enough images in pickled data. Found {len(data_pairs)}, need {NUM_IMAGES}")
selected_pairs = random.sample(data_pairs, NUM_IMAGES)

# Initialize GCS client
try:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
except Exception as e:
    print(f"Error initializing GCS client: {e}")
    exit(1)



# Function to process a batch of images
def process_batch(pairs, required_images=10):
    processed_data = []
    processed_count = 0
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    current_index = 0

    while processed_count < required_images and current_index < len(pairs):
        pair = pairs[indices[current_index]]
        image_path, _ = pair
        
        if image_path.startswith('gs://'):
            blob_name = image_path.replace(f'gs://{BUCKET_NAME}/', '')
            temp_file = None
            local_path = None
            for attempt in range(3):
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.close()
                    blob = bucket.blob(blob_name)
                    blob.download_to_filename(temp_file.name)
                    local_path = temp_file.name
                    break
                except Exception as e:
                    print(f"Error downloading image {image_path} (attempt {attempt+1}): {e}")
                    if temp_file and os.path.exists(temp_file.name):
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                    time.sleep(1)
            if not local_path:
                print(f"Failed to download image {image_path} after retries")
                current_index += 1
                continue
        else:
            local_path = image_path
        
        frame = cv2.imread(local_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            if local_path != image_path and os.path.exists(local_path):
                try:
                    os.unlink(local_path)
                except:
                    pass
            current_index += 1
            continue
        
        try:
            results = yolo_model.track(frame, persist=True, tracker="botsort.yaml")
        except Exception as e:
            print(f"Error running YOLO on image {image_path}: {e}")
            if local_path != image_path and os.path.exists(local_path):
                try:
                    os.unlink(local_path)
                except:
                    pass
            current_index += 1
            continue
        
        faces = []
        boxes = []
        for r in results:
            for box in r.boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (224, 224))
                        crop_preprocessed = preprocess_input(crop_resized.astype(np.float32))
                        faces.append(crop_preprocessed)
        
        if faces:
            faces_batch = np.array(faces)
            try:
                preds = mtl_model.predict(faces_batch, verbose=0)
            except Exception as e:
                print(f"Error running model prediction on image {image_path}: {e}")
                if local_path != image_path and os.path.exists(local_path):
                    try:
                        os.unlink(local_path)
                    except:
                        pass
                current_index += 1
                continue
            expr_preds = preds['expr']
            va_preds = preds['va']
            au_preds = preds['au']
            
            i = 0
            label, final_probs, O, emotions, expr_probs, va_pred = translate_outputs(expr_preds[i], va_preds[i], au_preds[i])
            au_labels_str = ", ".join([AU_LABELS[k] for k in O])
            top_3 = sorted(zip(emotions, final_probs), key=lambda x: x[1], reverse=True)[:3]
            top_3_str = ", ".join([f"{emotion}: {prob:.2f}" for emotion, prob in top_3])
            va_str = f"Valence: {va_pred[0]:.2f}, Arousal: {va_pred[1]:.2f}"
            expr_str = ", ".join([f"{em}: {prob:.2f}" for em, prob in zip(emotions, expr_probs)])
            
            display_image = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            image_name = os.path.basename(image_path)
            processed_data.append((display_image, label, au_labels_str, top_3_str, va_str, expr_str, image_name))
            processed_count += 1
        
        if local_path != image_path and os.path.exists(local_path):
            try:
                os.unlink(local_path)
            except:
                pass
        
        current_index += 1
    
    if processed_count < required_images:
        print(f"Warning: Only {processed_count} images processed successfully. Required {required_images}.")
    
    return processed_data

# Function to display images
def display_images(processed_data, fig, axes):
    for ax in axes.flat:
        ax.clear()
    
    for j, (image, label, au_labels_str, top_3_str, va_str, expr_str, image_name) in enumerate(processed_data[:10]):
        row, col = divmod(j, 5)
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f"{label}\n{image_name}", fontsize=8)
        ax.axis('off')
        ax.text(0.5, -0.2, au_labels_str, ha='center', va='top', transform=ax.transAxes, fontsize=6)
        ax.text(0.5, -0.3, f"Top 3: {top_3_str}", ha='center', va='top', transform=ax.transAxes, fontsize=6)
        ax.text(0.5, -0.4, va_str, ha='center', va='top', transform=ax.transAxes, fontsize=6)
        ax.text(0.5, -0.5, f"Raw Expr: {expr_str}", ha='center', va='top', transform=ax.transAxes, fontsize=6, wrap=True)
    
    plt.subplots_adjust(bottom=0.5, hspace=0.5)
    fig.canvas.draw()

# Callback for the Next button
def on_next_button_clicked(event, fig, axes):
    selected_pairs = random.sample(data_pairs, min(NUM_IMAGES * 2, len(data_pairs)))
    processed_data = process_batch(selected_pairs, required_images=NUM_IMAGES)
    display_images(processed_data, fig, axes)

# Set up interactive plot
plt.ion()
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.subplots_adjust(bottom=0.2)

# Create Next button
ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
next_button = Button(ax_button, 'Next')
next_button.on_clicked(lambda event: on_next_button_clicked(event, fig, axes))

# Initial batch
selected_pairs = random.sample(data_pairs, min(NUM_IMAGES * 2, len(data_pairs)))
processed_data = process_batch(selected_pairs, required_images=NUM_IMAGES)
display_images(processed_data, fig, axes)

# Keep the plot open
plt.show(block=True)

# # Clean up pickle file
# if os.path.exists(pickle_path):
#     try:
#         os.unlink(pickle_path)
#     except:
#         pass