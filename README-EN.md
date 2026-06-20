# FACE SIMILARITY SYSTEM: EIGENFACES VIA SVD (OPTIMIZED FUSION AND AGING VERSION)

A modern, lightweight implementation of the classic **Eigenfaces** facial recognition system (Turk & Pentland, 1991). The core engine utilizes **Principal Component Analysis (PCA)** executed via **Singular Value Decomposition (SVD)** to project high-dimensional facial images into a compact linear subspace (*eigenspace*).

The key enhancements in this version include **Sensor Fusion (Pixel, LBP, HOG)** and **Aging Vectors** for cross-age verification, pre-trained on three large datasets: **LFW**, **FG-NET**, and **IMSFD (Southeast Asian and Hijab Faces)**.

---

## Technical Architecture and Enhancements

1.  **Combined Eigenspace Training (LFW + FG-NET + IMSFD):**
    - **LFW:** Enriches the facial space with general expressions and poses.
    - **FG-NET:** Used to track age progression (ages 1–69 years).
    - **IMSFD (Indonesia Muslim Student Face Dataset):** Enriches the space with Southeast Asian facial characteristics and hijab variations.
2.  **Sensor Fusion (Multimodality):**
    - **Pixel PCA:** Captures rough facial silhouettes and global illumination.
    - **LBP PCA (Local Binary Patterns):** Captures local texture patterns (robust to extreme shadow and illumination changes).
    - **HOG PCA (Histogram of Oriented Gradients):** Captures stable facial contours.
3.  **Age Progression Compensation (Aging Vector):**
    - The model projects FG-NET onto the PCA subspace, divides the images into child (≤12 years) and adult (≥18 years) masks, and computes their mean difference as the direction of aging (aging_vector).
    - The system can simulate aging or de-aging on face vectors before comparison.

---

## Repository Structure (folder improved)

*   app.py: Streamlit frontend UI and visualization of preprocessing steps.
*   face_similarity.py: Main algebraic computation engine (LBP, HOG features, similarity fusion, and SVD decomposition).
*   pretrained_eigenspace.npz: Compressed training output payload (contains Pixel, LBP, HOG PCA matrices, and aging vectors).
*   lbfmodel.yaml: Pre-trained model parameters for landmark mapping (LBF Facemark).
*   requirements.txt & requirements.md: Environmental configuration requirements.

---

## Installation & Launching

### 1. Install Dependencies
Ensure Python 3.9 to 3.11 is installed. Run the following command in the workspace directory:
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit
Launch the local web server dashboard:
```bash
streamlit run app.py
```

---

## Loading the Pre-trained Model (.npz)

To run the application:
1.  Navigate to the **Train** tab in the Streamlit UI.
2.  Check the **Model Pra-Latih (.npz)** checkbox.
3.  The system will automatically locate `pretrained_eigenspace.npz` within the `improved/` folder.
4.  Click **Latih sekarang** to load the model instantly, view the Scree plots, and perform comparisons in the **Compare** or **Identify** tabs.
