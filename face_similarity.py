"""
face_similarity.py
==================
Sistem Deteksi Kemiripan Wajah berbasis Eigenfaces (PCA via SVD) - Versi Optimasi & Visualisasi
Mata Kuliah Aljabar Linear — Teknik Informatika

Penggunaan:
    python face_similarity.py             # menjalankan demo end-to-end
    from face_similarity import EigenfaceModel, compare_two_faces, recognize_face

Author : (John) + Claude + Antigravity
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# KONSTANTA GLOBAL & CACHING MODEL DETEKSI / ALIGNMENT
# ============================================================
IMG_SIZE: Tuple[int, int] = (100, 100)   # ukuran wajah seragam (W, H)

# Global cache untuk model
_FACE_CASCADE_CACHE = None
_FACEMARK_LBF_CACHE = None

LBF_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbfmodel.yaml")


def get_face_cascade() -> cv2.CascadeClassifier:
    """Helper untuk memuat Haar Cascade secara lazy-loading dan di-cache."""
    global _FACE_CASCADE_CACHE
    if _FACE_CASCADE_CACHE is None:
        xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE_CACHE = cv2.CascadeClassifier(xml_path)
        if _FACE_CASCADE_CACHE.empty():
            raise FileNotFoundError(
                f"Gagal memuat XML Haar Cascade dari OpenCV. Pastikan opencv-python terinstal dengan benar."
            )
    return _FACE_CASCADE_CACHE


def get_facemark_lbf():
    """Helper untuk memuat OpenCV Facemark LBF secara lazy-loading dan di-cache."""
    global _FACEMARK_LBF_CACHE
    if _FACEMARK_LBF_CACHE is None:
        if hasattr(cv2, 'face') and os.path.exists(LBF_MODEL_PATH):
            try:
                fm = cv2.face.createFacemarkLBF()
                fm.loadModel(LBF_MODEL_PATH)
                _FACEMARK_LBF_CACHE = fm
            except Exception as e:
                print(f"⚠️ Warning: Gagal menginisialisasi LBF Facemark: {e}")
    return _FACEMARK_LBF_CACHE


# ============================================================
# 1. ALIGNMENT & EXTRACTION FITUR
# ============================================================
def align_face_lbf(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = IMG_SIZE) -> Tuple[np.ndarray, bool]:
    """
    Supervised Descent Method (SDM) LBF Alignment.
    Memutar, menyesuaikan skala, dan mentranslasi wajah sehingga mata selalu jatuh pada koordinat piksel yang absolut.
    """
    facemark = get_facemark_lbf()
    if facemark is None:
        return gray_image, False

    x, y, w, h = bbox
    # Model Facemark memerlukan koordinat bounding box
    ok, landmarks = facemark.fit(gray_image, np.array([[x, y, w, h]]))
    if not ok or len(landmarks) == 0:
        return gray_image, False
        
    pts = landmarks[0][0]
    # Mata kiri: landmark indeks 36-41. Mata kanan: landmark indeks 42-47.
    left_eye = np.mean(pts[36:42], axis=0)
    right_eye = np.mean(pts[42:48], axis=0)
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    dist = np.sqrt(dx**2 + dy**2)
    # Jarak antar mata diset konstan 40% dari lebar gambar akhir
    desired_dist = target_size[0] * 0.40
    scale = desired_dist / max(dist, 1.0)
    
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Translasi: geser pusat mata ke koordinat X=50%, Y=35% dari gambar akhir
    t_x = target_size[0] * 0.50
    t_y = target_size[1] * 0.35
    M[0, 2] += (t_x - eye_center[0])
    M[1, 2] += (t_y - eye_center[1])
    
    aligned_face = cv2.warpAffine(gray_image, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned_face, True


def crop_face_inner(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = -0.08) -> np.ndarray:
    """Memotong wajah ke dalam (inner crop) untuk mengurangi pengaruh rambut & latar belakang."""
    x, y, w, h = bbox
    H, W = gray_image.shape
    x1 = max(0, x - int(w * padding))
    y1 = max(0, y - int(h * padding))
    x2 = min(W, x + w + int(w * padding))
    y2 = min(H, y + h + int(h * padding))
    return gray_image[y1:y2, x1:x2]


def extract_lbp_features(image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """Mengekstrak fitur Local Binary Patterns (LBP) untuk tekstur wajah (tahan cahaya)."""
    try:
        from skimage.feature import local_binary_pattern
        if image.dtype != np.uint8:
            img_uint8 = ((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img_uint8 = image
        lbp = local_binary_pattern(img_uint8, P=P, R=R, method="uniform")
        return lbp.flatten().astype(np.float64) / max(lbp.max(), 1.0)
    except Exception:
        # Fallback NumPy murni jika skimage tidak terinstal
        if image.dtype != np.uint8:
            img_uint8 = ((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img_uint8 = image
        H, W = img_uint8.shape
        lbp = np.zeros((H, W), dtype=np.uint8)
        angles = [2 * np.pi * p / P for p in range(P)]
        for y in range(R, H - R):
            for x in range(R, W - R):
                center = int(img_uint8[y, x])
                code = 0
                for p, angle in enumerate(angles):
                    nx = x + R * np.cos(angle)
                    ny = y - R * np.sin(angle)
                    nx_i, ny_i = int(round(nx)), int(round(ny))
                    nx_i = np.clip(nx_i, 0, W - 1)
                    ny_i = np.clip(ny_i, 0, H - 1)
                    if int(img_uint8[ny_i, nx_i]) >= center:
                        code |= 1 << p
                lbp[y, x] = code
        return lbp.flatten().astype(np.float64) / 255.0


def extract_hog_features(image: np.ndarray, orientations: int = 8, pixels_per_cell: Tuple[int, int] = (8, 8), cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """Mengekstrak fitur Histogram of Oriented Gradients (HOG) untuk geometri bentuk wajah."""
    try:
        from skimage.feature import hog
        if image.dtype != np.uint8:
            img_uint8 = ((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img_uint8 = image
        features = hog(
            img_uint8,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
        )
        return features.astype(np.float64)
    except Exception:
        # Fallback NumPy + Sobel jika skimage tidak terinstal
        if image.dtype != np.uint8:
            img = ((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            img = image.copy()
        img_float = img.astype(np.float64)
        Gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=1)
        Gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=1)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx) * (180.0 / np.pi) % 180.0
        H, W = img.shape
        cy, cx = pixels_per_cell
        n_cells_y = H // cy
        n_cells_x = W // cx
        histograms = []
        for y in range(n_cells_y):
            for x in range(n_cells_x):
                cell_mag = magnitude[y * cy : (y + 1) * cy, x * cx : (x + 1) * cx]
                cell_ori = orientation[y * cy : (y + 1) * cy, x * cx : (x + 1) * cx]
                hist, _ = np.histogram(cell_ori, bins=orientations, range=(0, 180), weights=cell_mag)
                norm = np.linalg.norm(hist)
                histograms.append(hist / (norm + 1e-6))
        return np.concatenate(histograms).astype(np.float64)


# ============================================================
# 2. PREPROCESSING & STEP VISUALIZATION
# ============================================================
class Preprocessor:
    """
    Konversi gambar mentah → vektor siap-PCA.
    """
    def __init__(self, image_size: Tuple[int, int] = (64, 64)):
        self.image_size = image_size            # (H, W)
        self.n_features = image_size[0] * image_size[1]

    @staticmethod
    def rgb_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
        """Konversi RGB → grayscale dengan bobot luminance ITU-R BT.601."""
        if img_rgb.ndim == 2:
            return img_rgb.astype(np.float64)
        weights = np.array([0.299, 0.587, 0.114])
        return img_rgb[..., :3] @ weights

    def load_image(self, path: str) -> np.ndarray:
        """Load satu gambar dari disk → grayscale, resize, ternormalisasi [0,1]."""
        img = Image.open(path)
        arr = np.array(img)
        gray = self.rgb_to_grayscale(arr)
        gray_img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        gray_img = gray_img.resize((self.image_size[1], self.image_size[0]))
        return np.array(gray_img, dtype=np.float64) / 255.0


def get_preprocessing_steps(image_path: str,
                            image_size: Optional[Tuple[int, int]] = None,
                            use_face_detection: bool = False,
                            equalize_hist: bool = False) -> dict:
    """
    Memproses gambar dan mengembalikan dictionary berisi output dari setiap tahapan.
    Mendukung LBF Face Alignment dan CLAHE.
    """
    image_size = image_size or IMG_SIZE
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Gambar tidak ditemukan atau rusak: {image_path}")

    steps = {}
    # 1. Original Image (RGB untuk ditampilkan di UI)
    steps["1_original"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    steps["2_grayscale"] = gray

    # 3. Face Detection & Alignment
    aligned_success = False
    if use_face_detection:
        face_cascade = get_face_cascade()
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) > 0:
            # Pilih wajah dengan area terbesar
            bbox = tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])
            
            # Coba LBF Face Alignment
            aligned_face, success = align_face_lbf(gray, bbox, image_size)
            if success:
                steps["3_cropped"] = aligned_face
                working_img = aligned_face
                aligned_success = True
            else:
                # Fallback: Crop Wajah Inner
                face_crop = crop_face_inner(gray, bbox)
                steps["3_cropped"] = face_crop
                working_img = face_crop
        else:
            # Fallback jika wajah tidak terdeteksi
            steps["3_cropped"] = gray.copy()
            working_img = gray
    else:
        steps["3_cropped"] = None
        working_img = gray

    # 4. Resize
    if working_img.shape[:2] != image_size:
        resized = cv2.resize(working_img, image_size, interpolation=cv2.INTER_AREA)
    else:
        resized = working_img
    steps["4_resized"] = resized

    # 5. Contrast Enhancement (CLAHE / Histogram Equalization)
    if equalize_hist:
        # Menggunakan CLAHE agar kontras lokal adaptif dan tidak membesarkan noise background
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        equalized = clahe.apply(resized)
        steps["5_equalized"] = equalized
        final = equalized
    else:
        steps["5_equalized"] = None
        final = resized

    # 6. Normalization & Flatten (Hasil Akhir)
    normalized = final.astype(np.float64) / 255.0
    steps["6_normalized"] = normalized
    steps["7_flattened"] = normalized.flatten()
    steps["aligned_success"] = aligned_success

    return steps


# ============================================================
# 3. EIGENFACES MODEL (PCA via SVD) - FUSION SUPPORT
# ============================================================
class EigenfaceModel:
    """
    Implementasi PCA berbasis Singular Value Decomposition dengan dukungan fusi LBP & HOG.
    """
    @property
    def feature_mode(self) -> str:
        return self.__dict__.get("_feature_mode", "pixel")

    @feature_mode.setter
    def feature_mode(self, value: str):
        self.__dict__["_feature_mode"] = value

    def __init__(self, n_components: int = 100, use_sklearn: bool = True,
                 random_state: int = 42, feature_mode: str = "pixel"):
        self.n_components = n_components
        self.use_sklearn = use_sklearn
        self.random_state = random_state
        self.feature_mode = feature_mode  # "pixel" atau "fusion"

        # Modality: Pixels (utama)
        self.mean_face_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None          # (k, n)
        self.singular_values_: Optional[np.ndarray] = None     # (k,)
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self._sklearn_pca: Optional[Union[PCA, IncrementalPCA]] = None

        # Aging Vectors (FGNET / AAF)
        self.aging_vector_pix: Optional[np.ndarray] = None
        self.aging_vector_lbp: Optional[np.ndarray] = None
        self.aging_vector_hog: Optional[np.ndarray] = None

        self.aging_vector_aaf_pix: Optional[np.ndarray] = None
        self.aging_vector_aaf_lbp: Optional[np.ndarray] = None
        self.aging_vector_aaf_hog: Optional[np.ndarray] = None

        # Modality: LBP (tekstur)
        self.mean_lbp_: Optional[np.ndarray] = None
        self.components_lbp_: Optional[np.ndarray] = None
        self.singular_values_lbp_: Optional[np.ndarray] = None
        self.explained_variance_ratio_lbp_: Optional[np.ndarray] = None
        self._sklearn_pca_lbp: Optional[PCA] = None

        # Modality: HOG (geometri rahang/tulang wajah)
        self.mean_hog_: Optional[np.ndarray] = None
        self.components_hog_: Optional[np.ndarray] = None
        self.singular_values_hog_: Optional[np.ndarray] = None
        self.explained_variance_ratio_hog_: Optional[np.ndarray] = None
        self._sklearn_pca_hog: Optional[PCA] = None

        self.image_shape: Optional[Tuple[int, int]] = None     # (H, W)

    def fit(self, X: np.ndarray,
            image_shape: Optional[Tuple[int, int]] = None) -> "EigenfaceModel":
        """X: (m, n) — melatih PCA pada dataset wajah."""
        max_components = min(X.shape[0], X.shape[1])
        if self.n_components > max_components:
            self.n_components = max_components

        # Simpan image_shape
        if image_shape is None:
            n = X.shape[1]
            h = int(np.sqrt(n))
            image_shape = (h, h) if h * h == n else (1, n)
        self.image_shape = image_shape

        # --- FIT PIXEL PCA ---
        self.mean_face_ = X.mean(axis=0)
        if self.use_sklearn:
            self._sklearn_pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                whiten=False,
                random_state=self.random_state,
            )
            self._sklearn_pca.fit(X)
            self.components_ = self._sklearn_pca.components_
            self.singular_values_ = self._sklearn_pca.singular_values_
            self.explained_variance_ratio_ = self._sklearn_pca.explained_variance_ratio_
        else:
            # SVD manual dengan NumPy
            X_c = X - self.mean_face_
            U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
            self.components_ = Vt[: self.n_components]
            self.singular_values_ = s[: self.n_components]
            total_var = (s ** 2).sum()
            self.explained_variance_ratio_ = (s[: self.n_components] ** 2) / total_var

        # --- FIT LBP & HOG PCA (Jika fusion diaktifkan) ---
        if self.feature_mode == "fusion":
            # Ekstrak LBP & HOG untuk data latihan
            X_lbp_list = []
            X_hog_list = []
            for row in X:
                img_2d = row.reshape(self.image_shape)
                X_lbp_list.append(extract_lbp_features(img_2d))
                X_hog_list.append(extract_hog_features(img_2d))
            
            X_lbp = np.array(X_lbp_list)
            X_hog = np.array(X_hog_list)

            # Fit LBP PCA
            self.mean_lbp_ = X_lbp.mean(axis=0)
            self._sklearn_pca_lbp = PCA(n_components=min(self.n_components, X_lbp.shape[0]), random_state=self.random_state)
            self._sklearn_pca_lbp.fit(X_lbp)
            self.components_lbp_ = self._sklearn_pca_lbp.components_
            self.singular_values_lbp_ = self._sklearn_pca_lbp.singular_values_
            self.explained_variance_ratio_lbp_ = self._sklearn_pca_lbp.explained_variance_ratio_

            # Fit HOG PCA
            self.mean_hog_ = X_hog.mean(axis=0)
            self._sklearn_pca_hog = PCA(n_components=min(self.n_components, X_hog.shape[0]), random_state=self.random_state)
            self._sklearn_pca_hog.fit(X_hog)
            self.components_hog_ = self._sklearn_pca_hog.components_
            self.singular_values_hog_ = self._sklearn_pca_hog.singular_values_
            self.explained_variance_ratio_hog_ = self._sklearn_pca_hog.explained_variance_ratio_

        return self

    def partial_fit(self, X: np.ndarray, 
                    image_shape: Optional[Tuple[int, int]] = None) -> "EigenfaceModel":
        """Melatih model secara bertahap menggunakan IncrementalPCA."""
        if self.image_shape is None:
            if image_shape is None:
                n = X.shape[1]
                h = int(np.sqrt(n))
                image_shape = (h, h) if h * h == n else (1, n)
            self.image_shape = image_shape

        if self._sklearn_pca is None or not isinstance(self._sklearn_pca, IncrementalPCA):
            self._sklearn_pca = IncrementalPCA(n_components=self.n_components)

        if X.shape[0] < self.n_components:
            repeats = (self.n_components // X.shape[0]) + 1
            X_padded = np.repeat(X, repeats, axis=0)[:self.n_components]
            self._sklearn_pca.partial_fit(X_padded)
        else:
            self._sklearn_pca.partial_fit(X)

        self.mean_face_ = self._sklearn_pca.mean_
        self.components_ = self._sklearn_pca.components_
        self.explained_variance_ratio_ = self._sklearn_pca.explained_variance_ratio_
        if hasattr(self._sklearn_pca, "singular_values_"):
            self.singular_values_ = self._sklearn_pca.singular_values_

        # PENTING: IncrementalPCA tidak mendukung LBP/HOG fusion secara berkala saat latihan bertahap
        # karena LBP/HOG memerlukan fitur ekstraksi dinamis. Pada parsial fit, mode fusion akan nonaktif.
        self.feature_mode = "pixel"

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyeksi intensitas piksel ke ruang PCA (utama)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_c = X - self.mean_face_
        return X_c @ self.components_.T

    def transform_fusion(self, X: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Mengekstrak dan memproyeksikan ketiga modalitas (Pixel, LBP, HOG)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 1. Proyeksi Pixel
        z_pix = self.transform(X)

        # 2. Proyeksi LBP & HOG (jika fusion didukung)
        z_lbp = None
        z_hog = None
        if self.feature_mode == "fusion" and self.components_lbp_ is not None:
            X_lbp_list = []
            X_hog_list = []
            for row in X:
                img_2d = row.reshape(self.image_shape)
                X_lbp_list.append(extract_lbp_features(img_2d))
                X_hog_list.append(extract_hog_features(img_2d))
            
            X_lbp = np.array(X_lbp_list)
            X_hog = np.array(X_hog_list)

            z_lbp = (X_lbp - self.mean_lbp_) @ self.components_lbp_.T
            z_hog = (X_hog - self.mean_hog_) @ self.components_hog_.T

        return {
            "pixel": z_pix,
            "lbp": z_lbp,
            "hog": z_hog
        }

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Rekonstruksi vektor representasi berdimensi rendah kembali ke piksel."""
        return Z @ self.components_ + self.mean_face_

    def get_eigenfaces(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Reshape komponen utama menjadi gambar (k, H, W)."""
        return self.components_.reshape(-1, *image_shape)


def load_pretrained_eigenspace_model(filepath: str) -> Optional[EigenfaceModel]:
    """Memuat model eigenspace dari file .npz milik teman Anda."""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath)
        n = int(data["n_samples"].item() if data["n_samples"].ndim == 0 else data["n_samples"][0])
        k = int(data["k_components"].item() if data["k_components"].ndim == 0 else data["k_components"][0])
        shape = tuple(int(x) for x in data["image_shape"])

        # Deteksi feature mode
        feature_mode = "fusion" if "mean_lbp" in data else "pixel"

        model = EigenfaceModel(n_components=k, feature_mode=feature_mode)
        model.mean_face_ = data["mean_face"]
        model.components_ = data["eigenfaces"]
        model.singular_values_ = data["singular_values"]
        model.explained_variance_ratio_ = data["explained_variance_pct"] / 100.0
        model.image_shape = shape

        if feature_mode == "fusion":
            model.mean_lbp_ = data["mean_lbp"]
            model.components_lbp_ = data["eigenfaces_lbp"]
            model.singular_values_lbp_ = data["singular_values_lbp"]
            model.explained_variance_ratio_lbp_ = data["explained_variance_pct"] / 100.0  # fallback

            if "mean_hog" in data:
                model.mean_hog_ = data["mean_hog"]
                model.components_hog_ = data["eigenfaces_hog"]
                model.singular_values_hog_ = data["singular_values_hog"]
                model.explained_variance_ratio_hog_ = model.explained_variance_ratio_ # fallback

        # Memuat aging vectors jika tersedia
        for mod in ["pix", "lbp", "hog"]:
            # FGNET (standard/default)
            key_fg = f"aging_vector_{mod}"
            if key_fg in data:
                setattr(model, f"aging_vector_{mod}", data[key_fg])
            elif f"aging_vector_fgnet_{mod}" in data:
                setattr(model, f"aging_vector_{mod}", data[f"aging_vector_fgnet_{mod}"])
                
            # AAF (Asian)
            key_aaf = f"aging_vector_aaf_{mod}"
            if key_aaf in data:
                setattr(model, f"aging_vector_aaf_{mod}", data[key_aaf])

        return model
    except Exception as e:
        print(f"⚠️ Error memuat pre-trained model: {e}")
        return None


# ============================================================
# 4. SIMILARITY ENGINE & METRICS FUSION
# ============================================================
def ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Perhitungan SSIM sederhana berbasis NumPy murni untuk perbandingan pixel-level."""
    a, b = img1.flatten(), img2.flatten()
    C1, C2 = 0.01**2, 0.03**2
    mu1, mu2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a), np.var(b)
    s12 = np.mean((a - mu1) * (b - mu2))
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return float(np.clip(num / den if den != 0 else 0, 0, 1))


class SimilarityEngine:
    """
    Mengukur kemiripan antara dua representasi wajah di ruang PCA.
    """
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a.flatten() - b.flatten()))

    @staticmethod
    def confidence_percent(cos_sim: float) -> float:
        return max(0.0, cos_sim) * 100.0

    @staticmethod
    def calibrated_confidence(cos_sim: float, threshold: float) -> float:
        if cos_sim <= threshold:
            denom = threshold - (-1.0)
            return 50.0 * (cos_sim - (-1.0)) / denom if denom > 1e-12 else 0.0
        denom = 1.0 - threshold
        return 50.0 + 50.0 * (cos_sim - threshold) / denom if denom > 1e-12 else 100.0

    @staticmethod
    def compute_all_metrics(
        z_a_dict: dict,
        z_b_dict: dict,
        face_a_disp: np.ndarray,
        face_b_disp: np.ndarray,
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.3,
        penalty_factor: float = 0.05
    ) -> Dict[str, float]:
        """
        Menghitung seluruh metrik kemiripan (Pixel, LBP, HOG) dan menggabungkannya.
        Juga mengabaikan PC 1-3 untuk menghitung jarak Euclidean guna membuang noise pencahayaan global.
        """
        w1_pix = z_a_dict["pixel"].flatten()
        w2_pix = z_b_dict["pixel"].flatten()

        cos_eigen = float(SimilarityEngine.cosine(w1_pix, w2_pix))

        # Pinalti Euclidean: membuang 3 PC pertama (PC1-PC3)
        w1_clean = w1_pix[3:] if len(w1_pix) > 3 else w1_pix
        w2_clean = w2_pix[3:] if len(w2_pix) > 3 else w2_pix
        euc_d = float(np.linalg.norm(w1_clean - w2_clean))

        # Konversi jarak Euclidean ke penalty score
        euc_penalty = min(penalty_factor, 0.001 * euc_d)
        score_pix = max(0.0, cos_eigen - euc_penalty)

        ssim_val = ssim_simple(face_a_disp, face_b_disp)
        cos_pixel = float(SimilarityEngine.cosine(face_a_disp, face_b_disp))

        result = {
            "cosine_similarity_eigenspace": round(cos_eigen, 4),
            "euclidean_distance_eigenspace": round(euc_d, 4),
            "ssim_pixel": round(ssim_val, 4),
            "cosine_similarity_pixel": round(cos_pixel, 4),
            "score_pix": round(score_pix, 4)
        }

        # Cek Fusi Fitur
        use_fusion = (
            z_a_dict.get("lbp") is not None
            and z_b_dict.get("lbp") is not None
            and z_a_dict.get("hog") is not None
            and z_b_dict.get("hog") is not None
            and (alpha + beta) > 0.0
        )

        if use_fusion:
            w1_lbp = z_a_dict["lbp"].flatten()
            w2_lbp = z_b_dict["lbp"].flatten()
            w1_hog = z_a_dict["hog"].flatten()
            w2_hog = z_b_dict["hog"].flatten()

            # LBP similarity
            cos_lbp = float(SimilarityEngine.cosine(w1_lbp, w2_lbp))
            wl1_clean = w1_lbp[3:] if len(w1_lbp) > 3 else w1_lbp
            wl2_clean = w2_lbp[3:] if len(w2_lbp) > 3 else w2_lbp
            d_lbp = float(np.linalg.norm(wl1_clean - wl2_clean))
            penalty_lbp = min(penalty_factor, 0.001 * d_lbp)
            score_lbp = max(0.0, cos_lbp - penalty_lbp)

            # HOG similarity
            cos_hog = float(SimilarityEngine.cosine(w1_hog, w2_hog))
            wh1_clean = w1_hog[3:] if len(w1_hog) > 3 else w1_hog
            wh2_clean = w2_hog[3:] if len(w2_hog) > 3 else w2_hog
            d_hog = float(np.linalg.norm(wh1_clean - wh2_clean))
            penalty_hog = min(penalty_factor, 0.001 * d_hog)
            score_hog = max(0.0, cos_hog - penalty_hog)

            # Hitung skor komposit tertimbang
            total_w = alpha + beta + gamma
            if total_w == 0:
                total_w = 1.0
            composite = np.clip(
                (alpha * score_lbp + beta * score_hog + gamma * score_pix) / total_w,
                0.0, 1.0
            )

            result.update({
                "cosine_lbp": round(cos_lbp, 4),
                "score_lbp": round(score_lbp, 4),
                "cosine_hog": round(cos_hog, 4),
                "score_hog": round(score_hog, 4),
                "composite_score": round(float(composite), 4),
                "feature_mode": "fusion"
            })
        else:
            result.update({
                "composite_score": round(float(np.clip(score_pix, 0.0, 1.0)), 4),
                "feature_mode": "pixel_only"
            })

        return result


# ============================================================
# 5. DOSEN-STYLE I/O (BACKWARD COMPATIBILITY)
# ============================================================
def load_face_image(path: str,
                    image_size: Optional[Tuple[int, int]] = None,
                    equalize_hist: bool = False) -> np.ndarray:
    """Membaca gambar wajah, crop/resize, normalisasi, dan flatten."""
    image_size = image_size or IMG_SIZE
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    if equalize_hist:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        resized = clahe.apply(resized)
    normalized = resized / 255.0
    return normalized.flatten()


def load_dataset(dataset_path: str,
                 image_size: Optional[Tuple[int, int]] = None,
                 use_face_detection: bool = False,
                 equalize_hist: bool = False,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Membaca seluruh dataset wajah dari folder."""
    image_size = image_size or IMG_SIZE
    loader = detect_and_crop_face if use_face_detection else load_face_image

    X: List[np.ndarray] = []
    labels: List[str] = []

    if not os.path.isdir(dataset_path):
        raise ValueError(f"Folder dataset tidak ditemukan: {dataset_path}")

    persons = sorted(d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d)))
    if verbose:
        print(f"  Memuat dataset dari: {dataset_path}")
        print(f"  Jumlah orang       : {len(persons)}")

    for person_name in persons:
        person_folder = os.path.join(dataset_path, person_name)
        n_loaded = 0
        for filename in sorted(os.listdir(person_folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_path = os.path.join(person_folder, filename)
                try:
                    vector = loader(image_path, image_size,
                                    equalize_hist=equalize_hist)
                    X.append(vector)
                    labels.append(person_name)
                    n_loaded += 1
                except ValueError as e:
                    if verbose:
                        print(f"    Lewati {image_path}: {e}")
        if verbose:
            print(f"    {person_name}: {n_loaded} foto")

    return np.array(X), np.array(labels)


def detect_and_crop_face(image_path: str,
                         image_size: Optional[Tuple[int, int]] = None,
                         equalize_hist: bool = False) -> np.ndarray:
    """Deteksi wajah, crop (LBF alignment / Haar fallback), resize, normalisasi, dan flatten."""
    image_size = image_size or IMG_SIZE
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        raise ValueError(f"Wajah tidak terdeteksi pada: {image_path}")

    bbox = max(faces, key=lambda f: f[2] * f[3])
    
    # Coba LBF Face Alignment
    aligned_face, success = align_face_lbf(gray, bbox, image_size)
    if success:
        face_crop = aligned_face
    else:
        face_crop = crop_face_inner(gray, bbox)
        
    face_resized = cv2.resize(face_crop, image_size)
    if equalize_hist:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)
    face_normalized = face_resized / 255.0
    return face_normalized.flatten()


# ============================================================
# 6. HIGH-LEVEL PIPELINES
# ============================================================
@dataclass
class CompareResult:
    cosine_similarity: float
    euclidean_distance: float
    confidence_percent: float
    calibrated_confidence_percent: float
    is_similar: bool
    threshold: float

    def pretty(self) -> str:
        status = "MIRIP ✅" if self.is_similar else "TIDAK MIRIP ❌"
        return (
            f"  cosine similarity      : {self.cosine_similarity: .4f}\n"
            f"  euclidean distance     : {self.euclidean_distance: .4f}\n"
            f"  raw confidence         : {self.confidence_percent:6.2f}%   (cos × 100)\n"
            f"  calibrated confidence  : {self.calibrated_confidence_percent:6.2f}%   (50% = threshold)\n"
            f"  threshold              : {self.threshold:.4f}\n"
            f"  keputusan              : {status}"
        )


@dataclass
class RecognizeResult:
    best_label: str
    best_similarity: float
    best_confidence_percent: float
    top_k_labels: List[str]
    top_k_similarities: List[float]
    top_k_indices: List[int]
    threshold: float

    def __iter__(self):
        yield self.best_label
        yield self.best_similarity

    def pretty(self) -> str:
        lines = [
            f"  best label   : {self.best_label}",
            f"  similarity   : {self.best_similarity:.4f}",
            f"  confidence   : {self.best_confidence_percent:.2f}%",
            f"  threshold    : {self.threshold:.4f}",
            f"  top-{len(self.top_k_labels)} match :",
        ]
        for lbl, s in zip(self.top_k_labels, self.top_k_similarities):
            lines.append(f"      {lbl:20s} sim = {s:.4f}")
        return "\n".join(lines)


def calibrate_threshold(model: "EigenfaceModel", X: np.ndarray, y: np.ndarray,
                        max_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Otomatis estimasi threshold optimal dari distribusi similarity di data latih.
    """
    rng = np.random.default_rng(seed)
    Z = model.transform(X)
    same_sims, diff_sims = [], []
    n = len(y)

    unique_labels = np.unique(y)
    label_to_indices = {label: np.where(y == label)[0] for label in unique_labels}
    valid_same_people = [lbl for lbl, idxs in label_to_indices.items() if len(idxs) > 1]
    half_pairs = max_pairs // 2

    # 1. Pasangan sama
    if len(valid_same_people) > 0:
        for _ in range(half_pairs):
            selected_label = rng.choice(valid_same_people)
            indices = label_to_indices[selected_label]
            i, j = rng.choice(indices, size=2, replace=False)
            s = SimilarityEngine.cosine(Z[i], Z[j])
            same_sims.append(s)
    else:
        print("  ⚠️  Tidak ditemukan orang dengan >= 2 foto. Kalibrasi disarankan minimal memiliki 2 foto per subjek.")
        return {
            "threshold": 0.5,
            "balanced_accuracy": 0.5,
            "same_mean": 0.0,
            "same_std": 0.0,
            "diff_mean": 0.0,
            "diff_std": 0.0,
            "n_same": 0,
            "n_diff": 0
        }

    # 2. Pasangan berbeda
    for _ in range(half_pairs):
        lbl1, lbl2 = rng.choice(unique_labels, size=2, replace=False)
        i = rng.choice(label_to_indices[lbl1])
        j = rng.choice(label_to_indices[lbl2])
        s = SimilarityEngine.cosine(Z[i], Z[j])
        diff_sims.append(s)

    same_sims, diff_sims = np.array(same_sims), np.array(diff_sims)

    candidates = np.linspace(-1, 1, 401)
    best_thr, best_acc = 0.5, 0.0
    for thr in candidates:
        tp = np.mean(same_sims >= thr)
        tn = np.mean(diff_sims < thr)
        acc = 0.5 * (tp + tn)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    return {
        "threshold": best_thr,
        "balanced_accuracy": best_acc,
        "same_mean": float(same_sims.mean()) if len(same_sims) > 0 else 0.0,
        "same_std": float(same_sims.std()) if len(same_sims) > 0 else 0.0,
        "diff_mean": float(diff_sims.mean()) if len(diff_sims) > 0 else 0.0,
        "diff_std": float(diff_sims.std()) if len(diff_sims) > 0 else 0.0,
        "n_same": len(same_sims),
        "n_diff": len(diff_sims),
    }


def _resolve_image_size(pca) -> Optional[Tuple[int, int]]:
    return getattr(pca, "image_shape", None)


def compare_two_faces(face_a: np.ndarray, face_b: np.ndarray,
                      model, threshold: float = 0.80) -> CompareResult:
    if face_a.ndim == 1:
        face_a = face_a.reshape(1, -1)
    if face_b.ndim == 1:
        face_b = face_b.reshape(1, -1)
    z_a = model.transform(face_a)
    z_b = model.transform(face_b)
    cos_sim = SimilarityEngine.cosine(z_a, z_b)
    eucl = SimilarityEngine.euclidean(z_a, z_b)
    conf = SimilarityEngine.confidence_percent(cos_sim)
    calib_conf = SimilarityEngine.calibrated_confidence(cos_sim, threshold)
    return CompareResult(
        cosine_similarity=cos_sim,
        euclidean_distance=eucl,
        confidence_percent=conf,
        calibrated_confidence_percent=calib_conf,
        is_similar=cos_sim >= threshold,
        threshold=threshold,
    )


def compare_faces(image_path_1: str, image_path_2: str,
                  pca, threshold: float = 0.80,
                  use_face_detection: bool = False) -> CompareResult:
    image_size = _resolve_image_size(pca)
    loader = detect_and_crop_face if use_face_detection else load_face_image
    face_1 = loader(image_path_1, image_size)
    face_2 = loader(image_path_2, image_size)
    return compare_two_faces(face_1, face_2, pca, threshold)


def _recognize_from_vector(face_vector: np.ndarray, pca, X_pca: np.ndarray,
                           labels, threshold: float = 0.80,
                           top_k: int = 5) -> RecognizeResult:
    z_q = pca.transform(face_vector.reshape(1, -1))
    similarities = cosine_similarity(z_q, X_pca)[0]
    top_idx = np.argsort(similarities)[::-1][:top_k]
    best_idx = int(top_idx[0])
    best_sim = float(similarities[best_idx])

    labels_arr = np.asarray(labels)
    return RecognizeResult(
        best_label=str(labels_arr[best_idx]) if best_sim >= threshold else "Tidak dikenal",
        best_similarity=best_sim,
        best_confidence_percent=SimilarityEngine.confidence_percent(best_sim),
        top_k_labels=[str(labels_arr[i]) for i in top_idx],
        top_k_similarities=[float(s) for s in similarities[top_idx]],
        top_k_indices=[int(i) for i in top_idx],
        threshold=threshold,
    )


def recognize_face(image_path: str, pca, X_pca: np.ndarray,
                   labels, threshold: float = 0.80,
                   top_k: int = 5,
                   use_face_detection: bool = False) -> RecognizeResult:
    image_size = _resolve_image_size(pca)
    loader = detect_and_crop_face if use_face_detection else load_face_image
    face = loader(image_path, image_size)
    return _recognize_from_vector(face, pca, X_pca, labels, threshold, top_k)


# ============================================================
# 7. DATA LOADERS & AUGMENTATION
# ============================================================
def _resize_image_batch(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    W, H = target_size
    if images.shape[1:3] == (H, W):
        return images.astype(np.float64) if images.dtype != np.float64 else images

    def _resize_one(img):
        if img.ndim == 3:
            img = img.mean(axis=2)
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.resize(img_uint8, (W, H)).astype(np.float64) / 255.0

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_resize_one, images))
    return np.stack(results)


def _equalize_hist_batch(images: np.ndarray) -> np.ndarray:
    def _equalize_one(img):
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(img_uint8).astype(np.float64) / 255.0

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_equalize_one, images))
    return np.stack(results)


def load_olivetti(image_size: Optional[Tuple[int, int]] = None,
                  equalize_hist: bool = False):
    data = fetch_olivetti_faces(shuffle=False)
    images = data.images
    if image_size is not None:
        images = _resize_image_batch(images, image_size)
    if equalize_hist:
        images = _equalize_hist_batch(images)
    X = images.reshape(len(images), -1)
    return X, data.target, images.shape[1:]


def load_lfw(min_faces_per_person: int = 20,
             image_size: Optional[Tuple[int, int]] = None,
             equalize_hist: bool = False):
    data = fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                            resize=0.5, color=False)
    images = data.images
    if image_size is not None:
        images = _resize_image_batch(images, image_size)
    if equalize_hist:
        images = _equalize_hist_batch(images)
    X = images.reshape(len(images), -1)
    target_names = data.target_names
    labels = np.array([target_names[i] for i in data.target])
    return X, labels, images.shape[1:], target_names


def load_large_dataset_in_batches(dataset_path: str, image_size: Tuple[int, int], batch_size: int = 500, equalize_hist: bool = False):
    """Generator memuat dataset secara berkala untuk efisiensi RAM."""
    X_batch = []
    labels_batch = []
    
    persons = sorted(d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)))
    
    for person_name in persons:
        person_folder = os.path.join(dataset_path, person_name)
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                try:
                    vector = load_face_image(image_path, image_size, equalize_hist=equalize_hist)
                    X_batch.append(vector)
                    labels_batch.append(person_name)
                    
                    if len(X_batch) == batch_size:
                        yield np.array(X_batch), np.array(labels_batch)
                        X_batch, labels_batch = [], []
                except Exception:
                    continue
                    
    if X_batch:
        yield np.array(X_batch), np.array(labels_batch)


def augment_horizontal_flip(X: np.ndarray, image_shape: Tuple[int, int],
                            labels: Optional[np.ndarray] = None):
    m, n = X.shape
    H, W = image_shape
    X_imgs = X.reshape(m, H, W)
    X_flipped = X_imgs[:, :, ::-1].reshape(m, n)
    X_aug = np.vstack([X, X_flipped])
    if labels is not None:
        labels_aug = np.concatenate([labels, labels])
        return X_aug, labels_aug
    return X_aug


def split_per_person(X: np.ndarray, y: np.ndarray, n_train_per_person: int = 7,
                     seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for person in np.unique(y):
        idx = np.where(y == person)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:n_train_per_person])
        test_idx.extend(idx[n_train_per_person:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


if __name__ == "__main__":
    print("face_similarity.py terintegrasi dengan LBF Face Alignment, Fusion LBP/HOG, dan pre-trained loader.")
