import os
import random
import pickle
from tqdm import tqdm
from PIL import Image
from google.cloud import storage
import numpy as np

# Set up GCS client (ensure credentials are configured locally)
client = storage.Client()
bucket_name = 'facial-expressions_bucket'
bucket = client.get_bucket(bucket_name)
output_prefix = 'Faces Dataset/fane/'

# Define input directory (adjust based on your local Kaggle dataset path)
input_dir = '/kaggle/input/fane-facial-expressions-and-emotion-dataset/fane_data'  # Replace with your actual path, e.g.

# Define emotion mappings and labels (based on FANE folder structure)
emotions = {
    'angry': {
        'expr': 1,
        'va': [-0.6, 0.7],  # Approximate valence-arousal
        'au': [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # AU4,5,17 for anger
    },
    'disgust': {
        'expr': 2,
        'va': [-0.7, 0.4],  # Approximate
        'au': [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]  # AU9,15,17
    },
    'fear': {
        'expr': 3,
        'va': [-0.5, 0.8],  # Approximate
        'au': [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]  # AU1,2,4,5,20,26
    },
    'sad': {
        'expr': 5,
        'va': [-0.8, -0.3],  # Approximate
        'au': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # AU1,4,15
    }
}

# List of desired folders
selected_emotions = ['angry', 'disgust', 'fear', 'sad']

# Process and upload images
fane_pairs = []
for emotion in selected_emotions:
    emotion_dir = os.path.join(input_dir, emotion)
    if not os.path.exists(emotion_dir):
        print(f"Warning: Directory {emotion_dir} not found. Skipping.")
        continue
    
    output_subprefix = f"{output_prefix}{emotion}/"
    print(f"Processing {emotion} images...")
    
    for filename in tqdm(os.listdir(emotion_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open and validate image
            img_path = os.path.join(emotion_dir, filename)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if file is a valid image
                    img = Image.open(img_path).convert('RGB')  # Ensure RGB format
                    
                    # Upload to GCS
                    blob_name = f"{output_subprefix}{filename}"
                    blob = bucket.blob(blob_name)
                    with io.BytesIO() as img_byte_arr:
                        img.save(img_byte_arr, format='JPEG')
                        img_byte_arr.seek(0)
                        blob.upload_from_file(img_byte_arr, content_type='image/jpeg')
                        print(f"Uploaded {blob_name}")
                    
                    # Create pair with GCS path
                    gs_path = f"gs://{bucket_name}/{blob_name}"
                    labels = emotions[emotion]
                    fane_pairs.append((gs_path, labels))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

# Shuffle for randomness
random.shuffle(fane_pairs)

# Pickle the FANE pairs
pickle_path = 'fane_mtl_pairs.pkl'  # Save locally first
with open(pickle_path, 'wb') as f:
    pickle.dump(fane_pairs, f)
print(f"Saved {len(fane_pairs)} FANE pairs to {pickle_path}")

# Optional: Move to Google Drive (manual or via Colab later)
# To move manually: Copy fane_mtl_pairs.pkl to /content/drive/My Drive/