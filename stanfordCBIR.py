"""
Stanford Online Products - Toy CBIR System
============================================
Assignment: Non-Textual Information Retrieval
Based on: toyCBIR.py (Prof. Raúl Alonso Calvo)

Application domain: E-commerce product visual search
- Query: photo of a product → retrieve visually similar products
- Dataset: Stanford Online Products (bicycles, chairs, lamps, mugs, etc.)
  FTP: ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip

Descriptor pipeline (Early Fusion):
  A. CNN ResNet50         → 2048 dim  (semantic)
  B. HSV Color Histogram  →  128 dim  (color - boosted for e-commerce)
  C. HOG                  →  324 dim  (structural shape)
  D. LBP Texture          →   59 dim  (surface material)
  ─────────────────────────────────────
  Total                   → 2559 dim

Each block is L2-normalized independently before weighted concatenation.
This prevents any single descriptor from dominating the L2 distance.
"""

import os
import cv2
import numpy as np
import nmslib
import pickle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


# ─────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────

def download_stanford_dataset(dest_dir="./Stanford_Online_Products"):
    """
    Download Stanford Online Products via FTP.
    Only runs if dataset is not already present.
    """
    import ftplib
    import zipfile

    if os.path.isdir(dest_dir):
        print(f"Dataset already at: {dest_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "Stanford_Online_Products.zip")

    print("Connecting to Stanford FTP...")
    ftp = ftplib.FTP("cs.stanford.edu")
    ftp.login()
    ftp.cwd("/cs/cvgl")

    print("Downloading Stanford_Online_Products.zip (this is large ~2GB)...")
    with open(zip_path, "wb") as f:
        ftp.retrbinary("RETR Stanford_Online_Products.zip", f.write)
    ftp.quit()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)
    print("Done.")


def load_stanford_split(dataset_root, split="train", max_images=None):
    """
    Parse Stanford Online Products split file.

    File format (Ebay_train.txt / Ebay_test.txt):
        image_id  super_class_id  class_id  path
        1         1               1          bicycle_final/111085122871_0.jpg
        ...

    Returns: list of (full_path, class_id, super_class_id)
    - class_id:       fine-grained label  (~23k classes)
    - super_class_id: coarse label        (12 product categories)
    """
    split_file = os.path.join(dataset_root, f"Ebay_{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    samples = []
    with open(split_file, "r") as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            _, super_cls, cls_id, rel_path = parts[0], int(parts[1]), int(parts[2]), parts[3]
            full_path = os.path.join(dataset_root, rel_path)
            if os.path.exists(full_path):
                samples.append((full_path, cls_id, super_cls))

    if max_images:
        samples = samples[:max_images]

    print(f"Loaded {len(samples)} images from {split} split.")
    return samples


# ─────────────────────────────────────────────────────────────
# CBIR SYSTEM
# ─────────────────────────────────────────────────────────────

class StanfordCBIRSystem:
    """
    Content-Based Image Retrieval system for Stanford Online Products.

    IR model quadruple [D, Q, F, R(qi, dj)]:
      D  = image collection (Stanford Online Products)
      Q  = query images (product photos)
      F  = feature extraction pipeline (CNN + Color + HOG + LBP)
      R  = L2 distance over weighted concatenated descriptors
    """

    # Descriptor block dimensions
    DIM_CNN   = 2048
    DIM_COLOR = 128   # 64 H + 32 S + 32 V  (more H bins for product hue)
    DIM_HOG   = 324
    DIM_LBP   = 59    # uniform LBP histogram

    def __init__(self,
                 index_file="stanford_cbir.nmslib",
                 metadata_file="stanford_metadata.pkl",
                 weights=None):
        """
        Parameters
        ----------
        weights : dict
            Per-descriptor scalar weights applied after independent L2 normalization.
            Boosting 'cnn' and 'color' suits e-commerce retrieval:
            color determines product identity, CNN captures category semantics.
        """
        self.index_file    = index_file
        self.metadata_file = metadata_file
        self.image_paths   = []
        self.image_labels  = []   # class_id per indexed image

        # ── A. CNN BACKBONE ─────────────────────────────────────
        # ResNet50 pretrained on ImageNet.
        # Suitable for Stanford Online Products (everyday objects).
        # Alternative for fine-grained: EfficientNetB4, ViT-B/16.
        self.cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

        # ── DESCRIPTOR WEIGHTS (Weighted Early Fusion) ───────────
        # Applied after each block's independent L2 normalization.
        # Rationale for e-commerce:
        #   - CNN(1.0): strong semantic anchor (identifies product category)
        #   - Color(1.2): color is the primary shopping filter
        #   - HOG(0.5): shape matters less than appearance for similar products
        #   - LBP(0.4): secondary texture cue (metal vs fabric vs plastic)
        self.weights = weights or {
            "cnn":   1.0,
            "color": 1.2,
            "hog":   0.5,
            "lbp":   0.4,
        }

        self.dimension = self.DIM_CNN + self.DIM_COLOR + self.DIM_HOG + self.DIM_LBP

        # ── NMSLIB INDEX ─────────────────────────────────────────
        # HNSW graph with L2 distance.
        # Cosine similarity would be theoretically better for CNN embeddings,
        # but since blocks are L2-normalized and weighted, L2 ≈ cosine on unit vectors.
        self.index = nmslib.init(method="hnsw", space="l2")

        print(f"Feature dimension: {self.dimension} "
              f"(CNN:{self.DIM_CNN} + Color:{self.DIM_COLOR} "
              f"+ HOG:{self.DIM_HOG} + LBP:{self.DIM_LBP})")

    # ── DESCRIPTOR EXTRACTION ────────────────────────────────────

    def _extract_cnn(self, img_rgb_224):
        """
        A. ResNet50 semantic embedding (2048-dim).
        Captures high-level semantic content: 'this is a bicycle'.
        L2-normalized → unit vector on hypersphere.
        """
        x = image.img_to_array(img_rgb_224)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = self.cnn_model.predict(x, verbose=0).flatten()
        feat /= (np.linalg.norm(feat) + 1e-7)
        return feat

    def _extract_color_hsv(self, img_rgb_224):
        """
        B. HSV Color Histogram (128-dim) — 'smart' histogram descriptor.

        Why HSV over RGB?
          HSV separates chromatic information (H) from illumination (V),
          making it more robust to lighting changes in product photos.

        Why 64 bins for H?
          Products in e-commerce are differentiated primarily by hue.
          Fine-grained hue discrimination (e.g., red vs orange mugs) requires
          more bins than the original 32.

        Distance: L2 after normalization.
        Theoretically, Chi-Square distance is superior for histograms,
        but since we fuse descriptors into a single vector for NMSLIB,
        we normalize each block to unit-L2 so distances are commensurable.
        """
        img_hsv = cv2.cvtColor(img_rgb_224, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([img_hsv], [0], None, [64], [0, 180])  # boosted
        hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])
        feat = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        feat /= (np.linalg.norm(feat) + 1e-7)
        return feat

    def _extract_hog(self, img_gray_224):
        """
        C. HOG — Histogram of Oriented Gradients (324-dim).
        Captures global structural shape: silhouette of the product.
        Useful for distinguishing product categories (lamp vs chair).
        """
        feat = hog(img_gray_224, orientations=9,
                   pixels_per_cell=(32, 32),
                   cells_per_block=(2, 2),
                   feature_vector=True)
        feat /= (np.linalg.norm(feat) + 1e-7)
        return feat

    def _extract_lbp(self, img_gray_224):
        """
        D. LBP — Local Binary Patterns, uniform mode (59-dim).
        Captures micro-texture: distinguishes metallic kettles from
        fabric sofas, or shiny plastics from matte surfaces.
        Uniform patterns (P=8, R=1) are rotation-invariant and compact.

        Added vs original toyCBIR: replaces the 4-bin Canny/Sobel descriptor
        with a richer texture descriptor more suitable for product material
        discrimination.
        """
        lbp = local_binary_pattern(img_gray_224, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
        feat = hist.astype("float32")
        feat /= (np.linalg.norm(feat) + 1e-7)
        return feat

    def extract_features(self, img_path):
        """
        Full pipeline: image → weighted concatenated feature vector.

        Fusion strategy: EARLY FUSION (late fusion requires separate indexes).
        Each block is independently L2-normalized, then scaled by its weight,
        then concatenated. This keeps descriptor magnitudes comparable while
        allowing domain tuning via weights without retraining.
        """
        img = cv2.imread(img_path)
        if img is None:
            return None

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res  = cv2.resize(img_rgb, (224, 224))
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)

        feat_cnn   = self._extract_cnn(img_res)       * self.weights["cnn"]
        feat_color = self._extract_color_hsv(img_res) * self.weights["color"]
        feat_hog   = self._extract_hog(img_gray)      * self.weights["hog"]
        feat_lbp   = self._extract_lbp(img_gray)      * self.weights["lbp"]

        return np.concatenate([feat_cnn, feat_color, feat_hog, feat_lbp]).astype("float32")

    # ── INDEXING ─────────────────────────────────────────────────

    def index_stanford(self, dataset_root, split="train", max_images=None):
        """
        Index Stanford Online Products split.
        Stores class labels to enable Precision@K evaluation.
        """
        samples = load_stanford_split(dataset_root, split, max_images)
        features_list = []
        self.image_paths  = []
        self.image_labels = []

        for path, class_id, _ in tqdm(samples, desc="Extracting features"):
            feat = self.extract_features(path)
            if feat is not None:
                features_list.append(feat)
                self.image_paths.append(path)
                self.image_labels.append(class_id)

        self._build_index(features_list)

    def index_folder(self, folder_path, max_images=None):
        """Generic folder indexing (no class labels)."""
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if max_images:
            files = files[:max_images]

        features_list = []
        self.image_paths  = []
        self.image_labels = []

        for fname in tqdm(files, desc="Indexing"):
            path = os.path.join(folder_path, fname)
            feat = self.extract_features(path)
            if feat is not None:
                features_list.append(feat)
                self.image_paths.append(path)

        self._build_index(features_list)

    def _build_index(self, features_list):
        """
        Build HNSW approximate nearest-neighbor index.
        Parameters:
          M=16             → connectivity of the graph (higher = better recall, more RAM)
          efConstruction=200 → search width during build (higher = better recall, slower build)
          post=2           → post-processing passes
        """
        if not features_list:
            print("No features to index.")
            return

        data_matrix = np.array(features_list)
        self.index.addDataPointBatch(data_matrix)

        print("Building HNSW graph...")
        self.index.createIndex(
            {"M": 16, "post": 2, "efConstruction": 200},
            print_progress=True
        )

        self.index.saveIndex(self.index_file, save_data=True)
        with open(self.metadata_file, "wb") as f:
            pickle.dump({"paths": self.image_paths, "labels": self.image_labels}, f)

        print(f"Index saved. {len(self.image_paths)} images indexed.")

    def load_index(self):
        """Load pre-built index from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index.loadIndex(self.index_file, load_data=True)
            with open(self.metadata_file, "rb") as f:
                meta = pickle.load(f)
            self.image_paths  = meta.get("paths", [])
            self.image_labels = meta.get("labels", [])
            print(f"Index loaded: {len(self.image_paths)} images.")
            return True
        return False

    # ── SEARCH ───────────────────────────────────────────────────

    def search(self, query_path, top_k=5):
        """
        kNN search: query image → top-k nearest images by L2 distance.
        Returns: list of (path, distance) sorted by ascending distance.
        """
        query_feat = self.extract_features(query_path)
        if query_feat is None:
            return []
        indices, distances = self.index.knnQuery(query_feat, k=top_k)
        return [(self.image_paths[idx], distances[i]) for i, idx in enumerate(indices)]

    def search_from_array(self, img_bgr, top_k=5):
        """Search from an already-loaded OpenCV image (BGR)."""
        tmp = "/tmp/_cbir_query_tmp.jpg"
        cv2.imwrite(tmp, img_bgr)
        return self.search(tmp, top_k)

    # ── EVALUATION ───────────────────────────────────────────────

    def precision_at_k(self, query_path, query_label, k=5):
        """
        Precision@K: fraction of top-k results sharing the query's class label.
        Range [0, 1]. Used to evaluate retrieval quality per query.
        """
        results = self.search(query_path, top_k=k)
        if not self.image_labels or not results:
            return None

        # Build path→label lookup
        path_to_label = dict(zip(self.image_paths, self.image_labels))
        relevant = sum(1 for path, _ in results if path_to_label.get(path) == query_label)
        return relevant / k

    def evaluate(self, test_samples, k=5, n_queries=200):
        """
        Evaluate mean Precision@K over a random subset of test queries.

        Parameters
        ----------
        test_samples : list of (path, class_id, super_class_id)
        k            : number of retrieved results to evaluate
        n_queries    : number of random queries to sample
        """
        if n_queries < len(test_samples):
            test_samples = random.sample(test_samples, n_queries)

        precisions = []
        for path, class_id, _ in tqdm(test_samples, desc=f"Evaluating P@{k}"):
            if os.path.exists(path):
                p = self.precision_at_k(path, class_id, k=k)
                if p is not None:
                    precisions.append(p)

        mean_p = float(np.mean(precisions)) if precisions else 0.0
        print(f"\nMean Precision@{k} = {mean_p:.4f}  ({len(precisions)} queries)")
        return mean_p

    # ── VISUALIZATION ─────────────────────────────────────────────

    def visualize(self, query_path, results, save_path="cbir_results.png"):
        """Display query image and top-k retrieved results."""
        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

        # Query
        q_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
        axes[0].imshow(q_img)
        axes[0].set_title("QUERY", fontweight="bold", color="navy")
        axes[0].axis("off")

        path_to_label = dict(zip(self.image_paths, self.image_labels))

        # Results
        for i, (path, dist) in enumerate(results):
            r_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(r_img)
            label_str = ""
            if self.image_labels:
                lbl = path_to_label.get(path)
                label_str = f"\nClass: {lbl}"
            axes[i + 1].set_title(f"Rank {i + 1}\nDist: {dist:.3f}{label_str}", fontsize=9)
            axes[i + 1].axis("off")

        plt.suptitle("Stanford Online Products — CBIR Results", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_path}")

    def visualize_descriptor_analysis(self, img_path):
        """
        Show the image alongside its descriptor visualizations.
        Useful for the report: demonstrates what each descriptor captures.
        """
        img = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res  = cv2.resize(img_rgb, (224, 224))
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
        img_hsv  = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Original
        axes[0].imshow(img_res)
        axes[0].set_title("Original (224×224)")
        axes[0].axis("off")

        # HSV H channel
        axes[1].imshow(img_hsv[:, :, 0], cmap="hsv")
        axes[1].set_title("HSV - Hue channel")
        axes[1].axis("off")

        # HOG visualization
        from skimage.feature import hog as hog_vis
        _, hog_image = hog_vis(img_gray, orientations=9,
                               pixels_per_cell=(32, 32),
                               cells_per_block=(2, 2),
                               visualize=True)
        axes[2].imshow(hog_image, cmap="gray")
        axes[2].set_title("HOG structure")
        axes[2].axis("off")

        # LBP visualization
        lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
        axes[3].imshow(lbp, cmap="gray")
        axes[3].set_title("LBP texture")
        axes[3].axis("off")

        # Edge map (Canny) — kept for reference vs original code
        edges = cv2.Canny(img_gray, 100, 200)
        axes[4].imshow(edges, cmap="gray")
        axes[4].set_title("Canny edges")
        axes[4].axis("off")

        plt.suptitle(f"Descriptor analysis: {os.path.basename(img_path)}", fontweight="bold")
        plt.tight_layout()
        plt.savefig("descriptor_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN — USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATASET_ROOT = "./Stanford_Online_Products"

    # ── 1. Download dataset (only if needed) ─────────────────────
    # download_stanford_dataset(DATASET_ROOT)   # uncomment for first run

    # ── 2. Instantiate system with domain-tuned weights ──────────
    # E-commerce rationale:
    #   CNN catches "what is it?" (bicycle vs mug)
    #   Color catches "which variant?" (red mug vs blue mug)  ← boosted
    #   HOG catches silhouette shape
    #   LBP catches surface material
    cbir = StanfordCBIRSystem(weights={
        "cnn":   1.0,
        "color": 1.2,
        "hog":   0.5,
        "lbp":   0.4,
    })

    # ── 3. Index (or load pre-built index) ───────────────────────
    if not cbir.load_index():
        # max_images: limit for quick testing; remove for full dataset
        cbir.index_stanford(DATASET_ROOT, split="train", max_images=3000)

    # ── 4. Load test split for evaluation and queries ────────────
    test_samples = load_stanford_split(DATASET_ROOT, split="test")

    # ── 5. Evaluate Precision@5 ──────────────────────────────────
    cbir.evaluate(test_samples, k=5, n_queries=200)

    # ── 6. Visual demo on a sample query ─────────────────────────
    if test_samples:
        query_path, query_class, _ = test_samples[0]
        print(f"\nQuery: {query_path}  (class {query_class})")

        results = cbir.search(query_path, top_k=5)
        cbir.visualize(query_path, results)

        # Descriptor analysis for report
        cbir.visualize_descriptor_analysis(query_path)