# Content-Based Image Retrieval 
Developed as part of the Information Retrieval course (ETSIINF) this project implements a CBIR system designed to search and retrieve images similar to a query image from a database. 

The system is tested on a subset of the **Stanford Online Products** dataset.

## Key Features

The retrieval engine extracts a combined feature vector (Dimension: 2498) comprising:
*   **Deep Semantic Features (ResNet50)**: Uses a pre-trained ResNet50 (on ImageNet) to extract high-level semantic representations (2048 dimensions).
*   **Color Histograms (HSV)**: Captures color distribution using HSV histograms to handle lighting variations (96 dimensions).
*   **HOG (Histogram of Oriented Gradients)**: Encodes object shape and silhouette structure (324 dimensions).
*   **LBP (Local Binary Patterns)**: Captures local texture details (26 dimensions).
*   **Edge Features (Canny/Sobel)**: Detects structural edges (4 dimensions).

### Indexing & Search
*   **Efficient Indexing**: Uses **NMSLIB** (Non-Metric Space Library) with the HNSW (Hierarchical Navigable Small World) algorithm for fast approximate nearest neighbor search.
*   **Distance Metric**: Cosine Similarity for the main index, refined by Chi-Square distance for color re-ranking.

## Project Structure

*   `toyCBIR.py`: The core implementation containing the `ToyCBIRSystem` class. It handles:
    *   Feature extraction pipeline.
    *   Index creation and management.
    *   Search querying and result visualization.
    *   Performance evaluation (Recall@K, mAP).
*   `download_data.py`: Helper script to manage dataset downloading and extraction.
*   `create_subset.py`: Utility to generate a lightweight version (`mini_dataset`) of the massive Stanford Online Products dataset for development purposes.

## Installation

Ensure you have Python 3.8+ installed.

### Dependencies
Install the required libraries:

```bash
pip install opencv-python numpy nmslib tensorflow scikit-image matplotlib tqdm gdown
```

## Usage

### 1. Setup Data
The system requires image data in the `mini_dataset` directory.
*   If you have the full `Stanford_Online_Products.zip`, you can use `create_subset.py` to generate a balanced subset.
*   Alternatively, `toyCBIR.py` attempts to set up the data using `download_dataset()`.

### 2. Run the System
Execute the main script to start the indexing and evaluation process:

```bash
python toyCBIR.py
```

**Workflow:**
1.  **Index Building**: If no index exists, the system scans `mini_dataset/`, extracts features for all images, and builds the HNSW graph.
2.  **Persistence**: The index is saved as `image_index_v2.nmslib` and metadata as `metadata_v2.pkl`.
3.  **Evaluation**: The system automatically partitions data into train/test sets (if not already done) and prints retrieval metrics (Recall@1, @5, @10 and mAP).
4.  **Visualization**: Displays the query image alongside the top 5 retrieved results.

## Configuration

Feature weights can be adjusted in `toyCBIR.py` to prioritize different visual aspects:

```python
W_CNN   = 2.0   # Dominant semantic matching
W_COLOR = 1.5   # Critical for product color matching
W_HOG   = 1.0   # Shape/Silhouette
W_CANNY = 0.3   # Edge details
```

## Performance Evaluation
The system outputs standard IR metrics:
*   **Recall@K**: Probability that the relevant class is found within the top K results.
*   **mAP (mean Average Precision)**: Measures the quality of the ranked retrieval results.
