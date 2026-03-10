import os
import cv2
import numpy as np
import nmslib
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from download_data import download_dataset, DATASET_DIR
import random
import glob


class ToyCBIRSystem:
    def __init__(self, index_file="image_index_v2.nmslib", metadata_file="metadata_v2.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.image_paths = []
        
        # 1. CNN Model (ResNet50) - 2048 dim embeddings
        self.cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Dimensions : 2048 (CNN) + 96 (Color) + 324 (HOG) + 4 (Canny/Sobel) = 2472
        self.dimension = 2472 
        
        # Init NMSLIB index HNSW and L2 distance
        self.index = nmslib.init(method='hnsw', space='l2') #, data_type=nmslib.DataType.FLOAT)

    def extract_features(self, img_path):
        """Extrae descriptores: CNN, Color HSV, HOG y Bordes Canny."""
        img = cv2.imread(img_path)
        if img is None: return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (224, 224))
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)

        # --- A. CNN FEATURES (Semantic) ---
        x = image.img_to_array(img_res)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat_cnn = self.cnn_model.predict(x, verbose=0).flatten()
        feat_cnn /= (np.linalg.norm(feat_cnn) + 1e-7) # normalize

        # --- B. COLOR HISTOGRAM (HSV) ---
        img_hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
        feat_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        feat_color /= (np.linalg.norm(feat_color) + 1e-7) # normalize

        # --- C. HOG (Global Structure) ---
        feat_hog = hog(img_gray, orientations=9, pixels_per_cell=(32, 32), 
                       cells_per_block=(2, 2), feature_vector=True)
        feat_hog /= (np.linalg.norm(feat_hog) + 1e-7) # normalize

        # --- D. CANNY + SOBEL (H/V Lines) ---
        # Detect borders: Canny
        edges = cv2.Canny(img_gray, 100, 200)
        # Calculate gradients for get line directions
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3) # Horizontal
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3) # Vertical
        
        # Classify borders in 4 bins (0°, 45°, 90°, 135°)
        angles = np.arctan2(sobely, sobelx) * 180 / np.pi
        angles = np.abs(angles)
        feat_canny = np.histogram(angles[edges > 0], bins=4, range=(0, 180))[0]
        feat_canny = feat_canny.astype('float32')
        feat_canny /= (np.linalg.norm(feat_canny) + 1e-7) # normalize

        # Concatenate normalized descriptors (Early Fusion)
        return np.concatenate([feat_cnn, feat_color, feat_hog, feat_canny]).astype('float32')

    def index_folder(self, folder_path):
        print(f"Indexing folder: {folder_path}...")
        features_list = []
        self.image_paths = []

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = [f for f in glob.glob(os.path.join(folder_path, "**/*"), recursive=True)
                if f.lower().endswith(valid_exts)]

        for path in tqdm(files):
            feat = self.extract_features(path)
            if feat is not None:
                features_list.append(feat)
                self.image_paths.append(path)

        if not features_list: return

        data_matrix = np.array(features_list)
        self.index.addDataPointBatch(data_matrix)
        print("Building HNSW graph...")
        self.index.createIndex({'M': 16, 'post': 2, 'efConstruction': 200}, print_progress=True)
        self.index.saveIndex(self.index_file, save_data=True)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.image_paths, f)
        print(f"Done. {len(self.image_paths)} images indexed.")

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index.loadIndex(self.index_file, load_data=True)
            with open(self.metadata_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            return True
        return False

    def search(self, query_path, top_k=5):
        # create the descriptors for input query image
        query_feat = self.extract_features(query_path)
        if query_feat is None: return []
        
        # k-NN search 1:N input query among all indexed images  
        indices, distances = self.index.knnQuery(query_feat, k=top_k)
        return [(self.image_paths[idx], distances[i]) for i, idx in enumerate(indices)]

    def visualize(self, query_path, results):
        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(20, 5))
        
        # Query
        axes[0].imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
        axes[0].set_title("IMAGEN CONSULTA")
        axes[0].axis('off')

        # Results
        for i, (path, dist) in enumerate(results):
            axes[i+1].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Rank {i+1}\nDist: {dist:.3f}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()

# --- USING THE TOY CBIR ---
if __name__ == "__main__":
    download_dataset()
    cbir = ToyCBIRSystem()

    if not cbir.load_index():
        cbir.index_folder(DATASET_DIR)

    query = random.choice(cbir.image_paths) 
    if os.path.exists(query):
        res = cbir.search(query, top_k=5)
        cbir.visualize(query, res)