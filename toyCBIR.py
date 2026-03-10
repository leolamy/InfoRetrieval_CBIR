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
from skimage.feature import local_binary_pattern
from download_data import download_dataset, DATASET_DIR
import random
import glob

# Weights Jusitfied for stanford online products dataset 
W_CNN   = 2.0   # sémantique dominante
W_COLOR = 1.5   # coloris produit critique
W_HOG   = 1.0   # silhouette
W_CANNY = 0.3   # peu discriminant fond blanc


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
        self.index = nmslib.init(method='hnsw', space='cosinesimil') #, data_type=nmslib.DataType.FLOAT)
        self.dimension = 2498

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

        # --- E. LBP (Local Texture) ---
        lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
        feat_lbp, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        feat_lbp = feat_lbp.astype('float32')
        feat_lbp /= (np.linalg.norm(feat_lbp) + 1e-7)
        
        # Classify borders in 4 bins (0°, 45°, 90°, 135°)
        angles = np.arctan2(sobely, sobelx) * 180 / np.pi
        angles = np.abs(angles)
        feat_canny = np.histogram(angles[edges > 0], bins=4, range=(0, 180))[0]
        feat_canny = feat_canny.astype('float32')
        feat_canny /= (np.linalg.norm(feat_canny) + 1e-7) # normalize

        return np.concatenate([
            feat_cnn   * W_CNN,
            feat_color * W_COLOR,
            feat_hog   * W_HOG,
            feat_canny * W_CANNY,
            feat_lbp   * 0.8   # texture matière
        ]).astype('float32')

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
        query_class = os.path.basename(os.path.dirname(query_path))
        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(20, 5))
        
        axes[0].imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"QUERY\n[{query_class}]")
        axes[0].axis('off')

        for i, (path, dist) in enumerate(results):
            ret_class = os.path.basename(os.path.dirname(path))
            is_correct = (ret_class == query_class)
            color = 'green' if is_correct else 'red'
            
            axes[i+1].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(
                f"Rank {i+1} | Dist: {dist:.3f}\n[{ret_class}]",
                color=color
            )
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"result_{os.path.basename(query_path)}.png", dpi=150)
        plt.show()

    def evaluate(self, test_images, k_values=[1, 5, 10]):
        results = {k: [] for k in k_values}

        for query_path in tqdm(test_images, desc="Evaluating"):
            query_class = os.path.basename(os.path.dirname(query_path))
            retrieved = self.search(query_path, top_k=max(k_values) + 1)
            retrieved = [(p, d) for p, d in retrieved if p != query_path]

            for k in k_values:
                top_k_classes = [
                    os.path.basename(os.path.dirname(p))
                    for p, _ in retrieved[:k]
                ]
                results[k].append(1 if query_class in top_k_classes else 0)

        print("\n--- Recall@K ---")
        for k in k_values:
            print(f"  Recall@{k}: {np.mean(results[k]):.4f}")
        return {k: np.mean(v) for k, v in results.items()}

# --- USING THE TOY CBIR ---
if __name__ == "__main__":
    download_dataset()
    cbir = ToyCBIRSystem()
    test_paths_file = "test_paths.pkl"          # pour stocker les chemins de test

    # --- Phase d'indexation (première exécution) ---
    if not cbir.load_index():
        # Récupérer tous les chemins d'images
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        all_files = glob.glob(os.path.join(DATASET_DIR, "**/*"), recursive=True)
        all_images = [f for f in all_files if f.lower().endswith(valid_exts)]
        random.shuffle(all_images)

        # Split train / test (80% / 20%)
        split = int(0.8 * len(all_images))
        train_images = all_images[:split]
        test_images  = all_images[split:]

        print(f"Indexation de {len(train_images)} images d'entraînement...")
        features_list = []
        cbir.image_paths = []   # ne contiendra que les images d'entraînement

        for path in tqdm(train_images):
            feat = cbir.extract_features(path)
            if feat is not None:
                features_list.append(feat)
                cbir.image_paths.append(path)

        if not features_list:
            print("Aucune feature extraite – abandon.")
            exit()

        # Construction et sauvegarde de l'index
        data_matrix = np.array(features_list)
        cbir.index.addDataPointBatch(data_matrix)
        print("Construction du graphe HNSW...")
        cbir.index.createIndex({'M': 16, 'post': 2, 'efConstruction': 200}, print_progress=True)
        cbir.index.saveIndex(cbir.index_file, save_data=True)
        with open(cbir.metadata_file, 'wb') as f:
            pickle.dump(cbir.image_paths, f)

        # Sauvegarde des chemins de test
        with open(test_paths_file, 'wb') as f:
            pickle.dump(test_images, f)

        print(f"Index terminé : {len(cbir.image_paths)} images en base.")

    # --- Phase d'évaluation (toutes exécutions) ---
    else:
        # Charger la liste des images de test
        if os.path.exists(test_paths_file):
            with open(test_paths_file, 'rb') as f:
                test_images = pickle.load(f)
        else:
            print("Fichier de test introuvable – exécutez d'abord sans index pour créer la séparation.")
            exit()

    # Vérifier qu'il y a assez d'images de test
    if len(test_images) == 0:
        print("Aucune image de test disponible.")
        exit()

    # Échantillonner 2000 images (ou moins si le jeu de test est plus petit)
    n_test = min(2000, len(test_images))
    test_sample = random.sample(test_images, n_test)

    # Évaluation
    cbir.evaluate(test_sample, k_values=[1, 5, 10])

    # Visualisation d'une requête aléatoire
    query = random.choice(test_sample)
    if os.path.exists(query):
        res = cbir.search(query, top_k=5)
        cbir.visualize(query, res)