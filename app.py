"""
app.py - Streamlit UI untuk Face Similarity (Eigenfaces PCA via SVD) - Versi Optimasi & Visualisasi Langkah
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_similarity import (
    EigenfaceModel,
    SimilarityEngine,
    IMG_SIZE,
    _recognize_from_vector,
    augment_horizontal_flip,
    calibrate_threshold,
    compare_two_faces,
    detect_and_crop_face,
    load_dataset,
    load_face_image,
    load_lfw,
    load_olivetti,
    get_preprocessing_steps,
    load_pretrained_eigenspace_model,
)

st.set_page_config(
    page_title="Face Similarity - Improved Version",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "model": None,
        "X": None,
        "X_pca": None,
        "labels": None,
        "image_shape": None,
        "image_size": IMG_SIZE,
        "threshold": 0.21,
        "n_components": 50,
        "equalize_hist": False,           # disimpan agar compare/identify konsisten
        "dataset_info": {},
        "calibration": None,
        "manual_photos": [],
        "compare_result": None,
        "compare_thumb_a": None,
        "compare_thumb_b": None,
        "compare_vec_a": None,
        "compare_vec_b": None,
        "compare_z_a": None,
        "compare_z_b": None,
        "compare_foto_a_name": "",
        "compare_foto_b_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# CACHED LOADERS — agar training kedua dgn param sama jadi instant
# ============================================================
@st.cache_data(show_spinner=False, max_entries=4)
def _cached_load_olivetti(image_size_tuple, equalize_hist):
    return load_olivetti(image_size=image_size_tuple, equalize_hist=equalize_hist)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_load_lfw(min_faces, image_size_tuple, equalize_hist):
    return load_lfw(min_faces_per_person=min_faces,
                    image_size=image_size_tuple,
                    equalize_hist=equalize_hist)


# ============================================================
# UTILITY
# ============================================================
def model_is_ready() -> bool:
    return st.session_state.model is not None


def reset_model():
    for k in ("model", "X", "X_pca", "labels", "image_shape",
              "dataset_info", "calibration", "compare_result", 
              "compare_thumb_a", "compare_thumb_b", "compare_vec_a", 
              "compare_vec_b", "compare_z_a", "compare_z_b"):
        st.session_state[k] = None
    st.session_state.dataset_info = {}
    st.session_state.manual_photos = []


def load_uploaded_image_as_array(uploaded_file, image_size,
                                 use_face_detection=False,
                                 equalize_hist=False):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if use_face_detection:
            vec = detect_and_crop_face(tmp_path, image_size,
                                       equalize_hist=equalize_hist)
        else:
            vec = load_face_image(tmp_path, image_size,
                                  equalize_hist=equalize_hist)
        img_bgr = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        return vec, img_rgb
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def extract_zip_to_dataset(uploaded_zip, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zf:
        zf.extractall(target_dir)
    entries = list(target_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        sub_entries = list(entries[0].iterdir())
        if any(p.is_dir() for p in sub_entries):
            return str(entries[0])
    return str(target_dir)


def serialize_model() -> bytes:
    payload = {
        "model": st.session_state.model,
        "X": st.session_state.X,
        "X_pca": st.session_state.X_pca,
        "labels": st.session_state.labels,
        "image_shape": st.session_state.image_shape,
        "image_size": st.session_state.image_size,
        "threshold": st.session_state.threshold,
        "equalize_hist": st.session_state.equalize_hist,
        "calibration": st.session_state.calibration,
        "dataset_info": st.session_state.dataset_info,
    }
    return pickle.dumps(payload)


def load_model_from_pickle(payload_bytes: bytes):
    payload = pickle.loads(payload_bytes)
    for k, v in payload.items():
        st.session_state[k] = v


def render_step_visualizer(image_file, use_face_detection, equalize_hist, title="Foto"):
    """Fungsi helper untuk merender visualisasi langkah-langkah preprocessing."""
    suffix = Path(image_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_file.getvalue())
        tmp_path = tmp.name
    try:
        steps = get_preprocessing_steps(
            tmp_path, 
            image_size=st.session_state.image_size,
            use_face_detection=use_face_detection,
            equalize_hist=equalize_hist
        )
        
        st.markdown(f"#### 🔍 Tahapan Preprocessing: {title}")
        
        # Buat kolom untuk setiap tahapan
        cols_count = 5 if use_face_detection or equalize_hist else 3
        cols = st.columns(cols_count)
        
        idx = 0
        # 1. Asli
        cols[idx].image(steps["1_original"], caption="1. Foto Asli (RGB)", use_container_width=True)
        idx += 1
        
        # 2. Grayscale
        cols[idx].image(steps["2_grayscale"], caption="2. Grayscale (Luminance)", use_container_width=True, clamp=True)
        idx += 1
        
        # 3. Crop Wajah (jika diaktifkan)
        if use_face_detection and steps["3_cropped"] is not None:
            aligned_lbl = "Aligned (LBF)" if steps.get("aligned_success") else "Cropped (Haar)"
            cols[idx].image(steps["3_cropped"], caption=f"3. {aligned_lbl}", use_container_width=True, clamp=True)
            idx += 1
            
        # 4. Resize
        cols[idx].image(steps["4_resized"], caption=f"4. Resize ({st.session_state.image_size[0]}x{st.session_state.image_size[1]})", use_container_width=True, clamp=True)
        idx += 1
        
        # 5. Equalization (jika diaktifkan)
        if equalize_hist and steps["5_equalized"] is not None:
            cols[idx].image(steps["5_equalized"], caption="5. CLAHE (Local Contrast)", use_container_width=True, clamp=True)
            idx += 1
            
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    st.sidebar.title("Face Similarity")
    st.sidebar.caption("Eigenfaces — PCA via SVD")

    st.sidebar.divider()
    st.sidebar.subheader("Database")

    if model_is_ready():
        info = st.session_state.dataset_info
        st.sidebar.success("Model siap")
        st.sidebar.markdown(f"""
- **Jumlah orang**: {info.get('n_persons', '?')}
- **Total foto**: {info.get('n_photos', '?')}
- **Image size**: {st.session_state.image_size[0]} × {st.session_state.image_size[1]}
- **Komponen**: {st.session_state.n_components}
- **Variansi terjelaskan**: {info.get('explained_variance', 0)*100:.2f}%
- **Feature Mode**: {getattr(st.session_state.model, "feature_mode", "pixel")}
- **Source**: {info.get('source', '?')}
""")
    else:
        st.sidebar.warning("Model belum dilatih")

    st.sidebar.divider()
    st.sidebar.subheader("Pengaturan Decision")

    # Nilai ambang batas kemiripan
    default_thr = 0.37 if getattr(st.session_state.model, "feature_mode", "pixel") == "fusion" else 0.21
    new_thr = st.sidebar.slider(
        "Threshold (Ambang Batas)",
        min_value=-1.0, max_value=1.0,
        value=float(st.session_state.threshold) if st.session_state.threshold != 0.21 else default_thr,
        step=0.01,
    )
    st.session_state.threshold = new_thr

    if st.session_state.calibration:
        c = st.session_state.calibration
        if st.sidebar.button("Pakai threshold optimal", use_container_width=True):
            st.session_state.threshold = c["threshold"]
            st.rerun()
        st.sidebar.caption(
            f"Auto-kalibrasi: τ\\* = {c['threshold']:.4f}  "
            f"(bal-acc = {c['balanced_accuracy']*100:.1f}%)"
        )

    st.sidebar.divider()
    st.sidebar.subheader("Model Pickle")

    if model_is_ready():
        st.sidebar.download_button(
            "Simpan model (.pkl)",
            data=serialize_model(),
            file_name="face_similarity_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    uploaded_model = st.sidebar.file_uploader(
        "Muat model (.pkl)", type=["pkl"], key="upload_model"
    )
    if uploaded_model is not None:
        try:
            load_model_from_pickle(uploaded_model.getvalue())
            st.sidebar.success("Model dimuat")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat: {e}")

    st.sidebar.divider()
    if st.sidebar.button("Reset", type="secondary", use_container_width=True):
        reset_model()
        st.rerun()


# ============================================================
# TAB TRAIN
# ============================================================
def render_train_tab():
    st.markdown("**Sumber dataset**")
    st.caption("Pilih satu atau gabungan beberapa sumber untuk melatih Eigenspace.")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        use_olivetti = st.checkbox("Olivetti (built-in)", value=False, key="src_olivetti")
    with col_b:
        use_lfw = st.checkbox("LFW (built-in)", value=False, key="src_lfw")
    with col_c:
        use_zip = st.checkbox("ZIP file", value=False, key="src_zip")
    with col_d:
        use_manual = st.checkbox("Upload manual", value=False, key="src_manual")
    with col_e:
        use_pretrained = st.checkbox("Model Pra-Latih (.npz)", value=False, key="src_pretrained")

    # ---- LFW config ----
    lfw_min_faces = None
    if use_lfw:
        with st.container(border=True):
            st.markdown("**Konfigurasi LFW**")
            lfw_min_faces = st.slider(
                "min_faces_per_person",
                min_value=5, max_value=100, value=20, step=5,
            )
            lfw_estimate = {5: "~5749 orang, ~13k foto", 10: "~158 orang, ~2370 foto",
                            20: "~62 orang, ~1140 foto", 30: "~34 orang, ~885 foto",
                            50: "~12 orang, ~570 foto", 70: "~7 orang, ~1288 foto",
                            100: "~5 orang, ~1140 foto"}
            nearest = min(lfw_estimate.keys(), key=lambda k: abs(k - lfw_min_faces))
            st.caption(f"Estimasi: {lfw_estimate[nearest]}")

    # ---- ZIP config ----
    zip_file = None
    if use_zip:
        with st.container(border=True):
            st.markdown("**Upload ZIP**")
            zip_file = st.file_uploader("File ZIP", type=["zip"], key="zip_upload")

    # ---- Manual config ----
    if use_manual:
        with st.container(border=True):
            st.markdown("**Upload manual**")
            mcol1, mcol2 = st.columns([2, 1])
            with mcol1:
                uploaded_photos = st.file_uploader(
                    "Upload foto",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key="manual_uploader",
                )
            with mcol2:
                label_name = st.text_input("Nama orang", key="manual_label")

            mbtn1, mbtn2 = st.columns([1, 1])
            with mbtn1:
                if st.button("Tambahkan ke koleksi") and uploaded_photos and label_name:
                    train_eq = st.session_state.get("_train_equalize_hist", False)
                    for uf in uploaded_photos:
                        try:
                            vec, _ = load_uploaded_image_as_array(
                                uf, st.session_state.get("_train_image_size", IMG_SIZE),
                                st.session_state.get("_train_face_detection", False),
                                equalize_hist=train_eq
                            )
                            st.session_state.manual_photos.append((label_name, vec))
                        except Exception as e:
                            st.error(f"Gagal memproses {uf.name}: {e}")
                    st.success(f"Ditambahkan {len(uploaded_photos)} foto untuk '{label_name}'")
            with mbtn2:
                if st.button("Kosongkan koleksi"):
                    st.session_state.manual_photos = []
                    st.rerun()

            if st.session_state.manual_photos:
                counts = Counter(p[0] for p in st.session_state.manual_photos)
                st.table({"Nama": list(counts.keys()),
                          "Jumlah": list(counts.values())})

    # ---- Pre-trained model loading ----
    if use_pretrained:
        with st.container(border=True):
            st.markdown("**Model Pra-Latih (.npz)**")
            st.caption("Memuat model eigenspace yang telah dilatih sebelumnya (Fusion LBP + HOG).")
            pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_eigenspace.npz")
            if os.path.exists(pretrained_path):
                st.info(f"File model ditemukan di: {pretrained_path}")
            else:
                st.error("File 'pretrained_eigenspace.npz' tidak ditemukan di folder 'improved/'. Silakan salin terlebih dahulu.")

    st.divider()
    st.markdown("**Pengaturan pelatihan**")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_components = st.number_input(
            "n_components (k)", min_value=2, max_value=500, value=50, step=1,
        )
    with col2:
        img_w = st.number_input("Lebar Gambar (W)", min_value=32, max_value=256,
                                value=100, step=4)
    with col3:
        img_h = st.number_input("Tinggi Gambar (H)", min_value=32, max_value=256,
                                value=100, step=4)
    image_size = (int(img_w), int(img_h))

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        use_face_detection = st.checkbox("Face detection (LBF alignment)", value=True)
    with pcol2:
        equalize_hist = st.checkbox("Histogram equalization (CLAHE)", value=True)
    with pcol3:
        do_augment = st.checkbox("Augmentasi horizontal flip", value=False)
    with pcol4:
        use_fusion = st.checkbox("Aktifkan Fusion (LBP + HOG)", value=True, help="Melatih PCA terpisah untuk LBP dan HOG.")

    # Simpan pengaturan pelatihan ke session state agar diakses konsisten saat upload manual
    st.session_state["_train_image_size"] = image_size
    st.session_state["_train_face_detection"] = use_face_detection
    st.session_state["_train_equalize_hist"] = equalize_hist

    st.divider()

    any_source = use_olivetti or use_lfw or use_zip or use_manual or use_pretrained
    if not any_source:
        st.info("Pilih minimal satu sumber dataset di atas.")
        if model_is_ready():
            render_post_training_section()
        return

    # ---- Tombol Latih ----
    if st.button("Latih sekarang", type="primary"):
        # Jika memilih pretrained model langsung
        if use_pretrained:
            pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_eigenspace.npz")
            with st.spinner("Memuat model pra-latih..."):
                model = load_pretrained_eigenspace_model(pretrained_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.image_shape = model.image_shape
                    st.session_state.image_size = (model.image_shape[1], model.image_shape[0])
                    st.session_state.n_components = model.n_components
                    st.session_state.equalize_hist = True  # CLAHE default
                    st.session_state.dataset_info = {
                        "n_persons": "?",
                        "n_photos": "Pra-latih",
                        "explained_variance": float(model.explained_variance_ratio_.sum()),
                        "source": "Pretrained Eigenspace (.npz)",
                        "equalize_hist": True,
                        "augmented": False,
                    }
                    # Dummy X/labels
                    st.session_state.X = model.mean_face_.reshape(1, -1)
                    st.session_state.labels = np.array(["mean_face"])
                    st.session_state.X_pca = model.transform(st.session_state.X)
                    st.success("Model pra-latih berhasil dimuat!")
                    st.rerun()
                else:
                    st.error("Gagal memuat model pra-latih. Pastikan file 'pretrained_eigenspace.npz' valid.")
                    return

        X_parts = []
        labels_parts = []
        source_descriptions = []

        if use_olivetti:
            with st.spinner("Memuat Olivetti..."):
                X_olv, y_olv, _ = _cached_load_olivetti(
                    image_size, equalize_hist
                )
                labels_olv = np.array([f"olv_{int(p):02d}" for p in y_olv])
                X_parts.append(X_olv)
                labels_parts.append(labels_olv)
                source_descriptions.append(
                    f"Olivetti ({len(X_olv)} foto, 40 orang)"
                )

        if use_lfw:
            with st.spinner(f"Memuat LFW (min_faces={lfw_min_faces})..."):
                try:
                    X_lfw, y_lfw, _, _ = _cached_load_lfw(
                        lfw_min_faces, image_size, equalize_hist
                    )
                    labels_lfw = np.array([f"lfw_{n}" for n in y_lfw])
                    X_parts.append(X_lfw)
                    labels_parts.append(labels_lfw)
                    source_descriptions.append(
                        f"LFW ({len(X_lfw)} foto, {len(np.unique(y_lfw))} orang)"
                    )
                except Exception as e:
                    st.error(f"Gagal memuat LFW: {e}")

        if use_zip:
            if zip_file is None:
                st.error("ZIP file belum di-upload")
                return
            with tempfile.TemporaryDirectory() as tmp_root:
                with st.spinner("Mengekstrak ZIP..."):
                    dataset_root = extract_zip_to_dataset(zip_file, tmp_root)
                with st.spinner("Memuat foto dari ZIP..."):
                    try:
                        X_zip, labels_zip = load_dataset(
                            dataset_root, image_size=image_size,
                            use_face_detection=use_face_detection,
                            equalize_hist=equalize_hist, verbose=False,
                        )
                    except Exception as e:
                        st.error(f"Gagal memuat ZIP: {e}")
                        return
                if len(X_zip) > 0:
                    X_parts.append(X_zip)
                    labels_parts.append(labels_zip)
                    source_descriptions.append(
                        f"ZIP ({len(X_zip)} foto, {len(np.unique(labels_zip))} orang)"
                    )

        if use_manual:
            if not st.session_state.manual_photos:
                st.error("Koleksi manual masih kosong")
                return
            labels_m = np.array([p[0] for p in st.session_state.manual_photos])
            X_m = np.array([p[1] for p in st.session_state.manual_photos])
            expected_n = image_size[0] * image_size[1]
            if X_m.shape[1] != expected_n:
                st.error(
                    f"Foto manual berukuran {X_m.shape[1]} piksel, "
                    f"tapi image_size pilihan = {image_size} ({expected_n} piksel). "
                    f"Kosongkan koleksi dan upload ulang."
                )
                return
            X_parts.append(X_m)
            labels_parts.append(labels_m)
            source_descriptions.append(
                f"Manual ({len(X_m)} foto, {len(np.unique(labels_m))} orang)"
            )

        if not X_parts:
            st.error("Tidak ada data yang berhasil dimuat")
            return

        X = np.vstack(X_parts)
        labels = np.concatenate(labels_parts)
        image_shape = (image_size[1], image_size[0])

        if do_augment:
            X, labels = augment_horizontal_flip(X, image_shape, labels)
            source_descriptions.append(f"+ flip aug (total {len(X)} foto)")

        with st.spinner("Melatih PCA..."):
            feature_mode = "fusion" if use_fusion else "pixel"
            model = EigenfaceModel(n_components=n_components, feature_mode=feature_mode).fit(
                X, image_shape=image_shape
            )
            X_pca = model.transform(X)

        st.session_state.model = model
        st.session_state.X = X
        st.session_state.X_pca = X_pca
        st.session_state.labels = labels
        st.session_state.image_shape = image_shape
        st.session_state.image_size = image_size
        st.session_state.equalize_hist = equalize_hist
        st.session_state.n_components = model.n_components
        st.session_state.dataset_info = {
            "n_persons": len(np.unique(labels)),
            "n_photos": len(X),
            "explained_variance": float(model.explained_variance_ratio_.sum()),
            "source": " + ".join(source_descriptions),
            "equalize_hist": equalize_hist,
            "augmented": do_augment,
        }
        st.success(
            f"Model dilatih: {len(X)} foto, {len(np.unique(labels))} orang"
        )
        st.rerun()

    if model_is_ready():
        render_post_training_section()


def render_post_training_section():
    st.divider()

    model = st.session_state.model
    image_shape = st.session_state.image_shape

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Mean Face**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(model.mean_face_.reshape(image_shape), cmap="gray")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("**Top 12 Eigenfaces**")
        eigs = model.get_eigenfaces(image_shape)
        n_show = min(12, len(eigs))
        cols_n = 4
        rows = (n_show + cols_n - 1) // cols_n
        fig, axes = plt.subplots(rows, cols_n, figsize=(cols_n * 1.6, rows * 1.7))
        axes = np.array(axes).flatten()
        for i in range(n_show):
            axes[i].imshow(eigs[i], cmap="gray")
            axes[i].set_title(f"PC {i+1}", fontsize=9)
            axes[i].axis("off")
        for j in range(n_show, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Scree Plot**")
    evr = model.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(range(1, len(evr) + 1), evr)
    axes[0].set_xlabel("Komponen ke-i")
    axes[0].set_ylabel("Explained variance ratio")
    axes[1].plot(range(1, len(cum) + 1), cum, marker="o", markersize=3)
    axes[1].axhline(0.95, ls="--", color="red", label="95%")
    axes[1].set_xlabel("Jumlah komponen")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.markdown("**Auto-kalibrasi threshold**")

    is_pretrained = (st.session_state.dataset_info.get("source") == "Pretrained Eigenspace (.npz)")

    if is_pretrained:
        st.info("💡 **Mode Pra-Latih:** Karena data latih asli tidak tersedia, Anda dapat memilih dataset evaluasi lain (seperti Olivetti) untuk mengestimasi threshold optimal pada eigenspace ini.")
        eval_dataset_choice = st.selectbox(
            "Pilih dataset evaluasi untuk kalibrasi:",
            ["Pilih...", "Olivetti Faces (400 foto, 40 orang)", "LFW (Labeled Faces in the Wild)"],
            key="eval_calib_dataset"
        )
        
        if eval_dataset_choice != "Pilih...":
            if st.button("Hitung threshold optimal (via Evaluasi)", type="primary"):
                with st.spinner("Memuat dataset evaluasi dan mengkalibrasi..."):
                    try:
                        if "Olivetti" in eval_dataset_choice:
                            X_eval, y_eval, _ = load_olivetti(
                                image_size=st.session_state.image_size,
                                equalize_hist=st.session_state.equalize_hist
                            )
                        else:
                            X_eval, y_eval, _, _ = load_lfw(
                                min_faces_per_person=20,
                                image_size=st.session_state.image_size,
                                equalize_hist=st.session_state.equalize_hist
                            )
                        
                        calib = calibrate_threshold(model, X_eval, y_eval)
                        st.session_state.calibration = calib
                        st.success(
                            f"Threshold optimal terkalibrasi: **{calib['threshold']:.4f}** "
                            f"(balanced accuracy: {calib['balanced_accuracy']*100:.2f}%)"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memuat dataset evaluasi untuk kalibrasi: {e}")
    else:
        n_labels = len(np.unique(st.session_state.labels))
        if n_labels < 2:
            st.warning("Butuh minimal 2 orang berbeda dari data latih langsung untuk kalibrasi.")
            return

        if st.button("Hitung threshold optimal"):
            with st.spinner("Mengkalibrasi..."):
                calib = calibrate_threshold(
                    model, st.session_state.X, st.session_state.labels
                )
                st.session_state.calibration = calib
                st.success(
                    f"Threshold optimal: {calib['threshold']:.4f}  "
                    f"(balanced accuracy: {calib['balanced_accuracy']*100:.2f}%)"
                )
                st.rerun()

    if st.session_state.calibration:
        c = st.session_state.calibration
        st.markdown(f"""
| Pasangan | Mean | Std | Jumlah |
|---|---|---|---|
| Orang sama | {c['same_mean']:.4f} | {c['same_std']:.4f} | {c['n_same']} |
| Orang berbeda | {c['diff_mean']:.4f} | {c['diff_std']:.4f} | {c['n_diff']} |

Threshold optimal: **{c['threshold']:.4f}**, balanced accuracy: **{c['balanced_accuracy']*100:.2f}%**
""")


# ============================================================
# TAB COMPARE
# ============================================================
def render_compare_tab():
    if not model_is_ready():
        st.warning("Model belum dilatih. Buka tab Train terlebih dahulu.")
        return

    # Inisialisasi default jika belum ada
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = None
        st.session_state.compare_thumb_a = None
        st.session_state.compare_thumb_b = None
        st.session_state.compare_vec_a = None
        st.session_state.compare_vec_b = None
        st.session_state.compare_z_a = None
        st.session_state.compare_z_b = None
        st.session_state.compare_z_a_orig = None
        st.session_state.compare_use_aging = True
        st.session_state.compare_aging_scale = 0.5
        st.session_state.compare_prob_asian = 0.80
        st.session_state.compare_foto_a_name = ""
        st.session_state.compare_foto_b_name = ""

    use_face_detection = st.checkbox(
        "Face detection (LBF alignment)", value=True, key="compare_face_detection",
    )

    # Deteksi vektor penuaan pada model
    model = st.session_state.model
    has_aging = (
        getattr(model, "aging_vector_pix", None) is not None
        or getattr(model, "aging_vector_aaf_pix", None) is not None
    )

    use_aging = False
    aging_scale = 0.5
    prob_asian = 0.80

    if has_aging:
        with st.expander("⏳ Kompensasi Perubahan Usia (Aging Vector)", expanded=True):
            use_aging = st.checkbox(
                "Aktifkan Injeksi Vektor Penuaan pada Foto A (Lama)",
                value=st.session_state.get("compare_use_aging", True),
                key="compare_use_aging_checkbox",
                help="Menyimulasikan proses penuaan pada foto masa kecil (Foto A) sebelum dibandingkan dengan wajah dewasa (Foto B)."
            )
            if use_aging:
                aging_scale = st.slider(
                    "Faktor Penuaan (Aging Scale)",
                    0.0, 2.0,
                    st.session_state.get("compare_aging_scale", 0.5),
                    0.1,
                    key="compare_aging_scale_slider",
                    help="Kekuatan pergeseran aljabar ke arah wajah tua."
                )
                prob_asian = st.slider(
                    "Probabilitas Wajah Asia (Asian Probability)",
                    0.0, 1.0,
                    st.session_state.get("compare_prob_asian", 0.80),
                    0.05,
                    key="compare_prob_asian_slider",
                    help="Proporsi pencampuran vektor penuaan Asia (AAF) vs Kaukasia (FGNET)."
                )

    # parameter fusi
    alpha = 0.3
    beta = 0.4
    gamma = 0.6
    penalty_factor = 0.05
    
    if getattr(st.session_state.model, "feature_mode", "pixel") == "fusion":
        with st.expander("⚖️ Bobot Sensor Fusion (LBP + HOG + Pixel)", expanded=True):
            alpha = st.slider("Bobot LBP (Tekstur)", 0.0, 2.0, 0.3, 0.1, help="Bobot kemiripan tekstur.")
            beta = st.slider("Bobot HOG (Bentuk)", 0.0, 2.0, 0.4, 0.1, help="Bobot kemiripan tepi geometri.")
            gamma = st.slider("Bobot Pixel (Intensitas)", 0.0, 2.0, 0.6, 0.1, help="Bobot kemiripan intensitas kasar.")
            penalty_factor = st.slider("Faktor Penalti Euclidean", 0.01, 0.20, 0.05, 0.01)
    else:
        with st.expander("⚙️ Parameter Jarak & Penalti", expanded=False):
            penalty_factor = st.slider("Faktor Penalti Euclidean", 0.01, 0.20, 0.05, 0.01, help="Penalti berbasis jarak Euclidean (tanpa PC1-PC3) untuk mereduksi false positive.")

    col1, col2 = st.columns(2)
    with col1:
        foto_a = st.file_uploader("Foto A (Lama / Anak-anak)", type=["jpg", "jpeg", "png"], key="compare_a")
    with col2:
        foto_b = st.file_uploader("Foto B (Baru / Dewasa)", type=["jpg", "jpeg", "png"], key="compare_b")

    # Reset hasil jika file foto berubah
    current_a_name = foto_a.name if foto_a else ""
    current_b_name = foto_b.name if foto_b else ""
    if (current_a_name != st.session_state.compare_foto_a_name or 
        current_b_name != st.session_state.compare_foto_b_name):
        st.session_state.compare_result = None
        st.session_state.compare_thumb_a = None
        st.session_state.compare_thumb_b = None
        st.session_state.compare_vec_a = None
        st.session_state.compare_vec_b = None
        st.session_state.compare_z_a = None
        st.session_state.compare_z_b = None
        st.session_state.compare_z_a_orig = None
        st.session_state.compare_foto_a_name = current_a_name
        st.session_state.compare_foto_b_name = current_b_name

    if foto_a and foto_b:
        # Tampilkan visualisasi langkah-langkah preprocessing
        with st.expander("🔍 Detail Langkah-langkah Preprocessing (Foto A & B)", expanded=True):
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                render_step_visualizer(foto_a, use_face_detection, st.session_state.equalize_hist, title="Foto A")
            with rcol2:
                render_step_visualizer(foto_b, use_face_detection, st.session_state.equalize_hist, title="Foto B")

        if st.button("Bandingkan Wajah", type="primary"):
            try:
                with st.spinner("Memproses perbandingan wajah..."):
                    vec_a, thumb_a = load_uploaded_image_as_array(
                        foto_a, st.session_state.image_size,
                        use_face_detection,
                        equalize_hist=st.session_state.equalize_hist)
                    vec_b, thumb_b = load_uploaded_image_as_array(
                        foto_b, st.session_state.image_size,
                        use_face_detection,
                        equalize_hist=st.session_state.equalize_hist)
                    
                    # Proyeksi wajah
                    z_a_dict = st.session_state.model.transform_fusion(vec_a)
                    z_b_dict = st.session_state.model.transform_fusion(vec_b)

                    # Simpan salinan proyeksi asli Foto A untuk visualisasi
                    z_a_dict_orig = {k: (v.copy() if v is not None else None) for k, v in z_a_dict.items()}

                    if use_aging:
                        # Terapkan injeksi vektor penuaan pada z_a_dict
                        for mod in ["pixel", "lbp", "hog"]:
                            attr_mod = "pix" if mod == "pixel" else mod
                            v_fg = getattr(st.session_state.model, f"aging_vector_{attr_mod}", None)
                            v_aaf = getattr(st.session_state.model, f"aging_vector_aaf_{attr_mod}", None)
                            
                            z_val = z_a_dict.get(mod)
                            if z_val is not None:
                                # default ke zero vector jika salah satu tidak tersedia
                                if v_fg is None:
                                    v_fg = np.zeros_like(z_val.flatten())
                                if v_aaf is None:
                                    v_aaf = np.zeros_like(z_val.flatten())
                                
                                # Hitung blended vector
                                blended_vector = prob_asian * v_aaf + (1.0 - prob_asian) * v_fg
                                
                                # Tambahkan ke z_val
                                z_a_dict[mod] = z_val + blended_vector.reshape(1, -1) * aging_scale

                    # Hitung semua metrik
                    metrics = SimilarityEngine.compute_all_metrics(
                        z_a_dict, z_b_dict,
                        vec_a.reshape(st.session_state.image_shape),
                        vec_b.reshape(st.session_state.image_shape),
                        alpha=alpha, beta=beta, gamma=gamma,
                        penalty_factor=penalty_factor
                    )

                    st.session_state.compare_result = metrics
                    st.session_state.compare_thumb_a = thumb_a
                    st.session_state.compare_thumb_b = thumb_b
                    st.session_state.compare_vec_a = vec_a
                    st.session_state.compare_vec_b = vec_b
                    st.session_state.compare_z_a = z_a_dict
                    st.session_state.compare_z_b = z_b_dict
                    st.session_state.compare_z_a_orig = z_a_dict_orig
                    st.session_state.compare_use_aging = use_aging
                    st.session_state.compare_aging_scale = aging_scale
                    st.session_state.compare_prob_asian = prob_asian
            except ValueError as e:
                st.error(str(e))
                return

        # Render hasil jika sudah di-compute
        if st.session_state.compare_result is not None:
            result = st.session_state.compare_result
            thumb_a = st.session_state.compare_thumb_a
            thumb_b = st.session_state.compare_thumb_b

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(thumb_a, caption="Wajah Ter-preprocess (Foto A)", use_container_width=True)
            with col_b:
                st.image(thumb_b, caption="Wajah Ter-preprocess (Foto B)", use_container_width=True)

            # Keputusan akhir
            is_similar = result["composite_score"] >= st.session_state.threshold
            calib_conf = SimilarityEngine.calibrated_confidence(result["composite_score"], st.session_state.threshold)

            if is_similar:
                st.success(
                    f"### MIRIP ✅\nCalibrated confidence: {calib_conf:.2f}%"
                )
            else:
                st.error(
                    f"### TIDAK MIRIP ❌\nCalibrated confidence: {calib_conf:.2f}%"
                )

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Composite Score (Final)", f"{result['composite_score']:.4f}")
            with mcol2:
                st.metric("Cosine Sim (Eigenspace)", f"{result['cosine_similarity_eigenspace']:.4f}")
            with mcol3:
                st.metric("Euclidean Dist (Eigenspace)", f"{result['euclidean_distance_eigenspace']:.4f}")
            with mcol4:
                st.metric("SSIM (Pixel-Level)", f"{result['ssim_pixel']:.4f}")

            # Jika menggunakan fusion, tampilkan breakdown metrik
            if "cosine_lbp" in result:
                st.markdown("##### 🔍 Detil Nilai Sensor Fusion")
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric("LBP Similarity (Tekstur)", f"{result['cosine_lbp']:.4f}")
                with fcol2:
                    st.metric("HOG Similarity (Bentuk)", f"{result['cosine_hog']:.4f}")
                with fcol3:
                    st.metric("Cosine Sim (Piksel Kasar)", f"{result['cosine_similarity_pixel']:.4f}")

            st.caption(f"Threshold keputusan yang digunakan: {st.session_state.threshold:.4f} (Calibrated 50%)")

            # --- SEKSI VISUALISASI MATEMATIS & ALJABAR INTERAKTIF ---
            with st.expander("📊 Analisis Aljabar Linear & Visualisasi Lanjutan", expanded=False):
                face_a_2d = st.session_state.compare_vec_a.reshape(st.session_state.image_shape)
                face_b_2d = st.session_state.compare_vec_b.reshape(st.session_state.image_shape)
                w1_pix = st.session_state.compare_z_a["pixel"].flatten()
                w2_pix = st.session_state.compare_z_b["pixel"].flatten()

                # 1. Pixel Matrix Representation
                st.markdown("#### 🔢 Representasi Matriks Piksel (Sub-Matriks 16x16 Kiri Atas)")
                col_mat1, col_mat2 = st.columns(2)
                with col_mat1:
                    st.markdown("**Matriks Foto A (0-1)**")
                    df1 = pd.DataFrame(face_a_2d[:16, :16])
                    st.dataframe(df1.style.background_gradient(cmap="gray", vmin=0.0, vmax=1.0), height=250)
                with col_mat2:
                    st.markdown("**Matriks Foto B (0-1)**")
                    df2 = pd.DataFrame(face_b_2d[:16, :16])
                    st.dataframe(df2.style.background_gradient(cmap="gray", vmin=0.0, vmax=1.0), height=250)

                # 2. SVD Reconstruction & Frobenius Norm
                st.markdown("#### 🌀 Rekonstruksi Wajah berbasis SVD ($A_k = U_k \\Sigma_k V_k^T$)")
                U1, S1, Vt1 = np.linalg.svd(face_a_2d, full_matrices=False)
                U2, S2, Vt2 = np.linalg.svd(face_b_2d, full_matrices=False)

                k_recon = st.slider("Jumlah Singular Value (k) untuk Rekonstruksi", 1, min(len(S1), 50), 10)
                
                def reconstruct_svd(U, S, Vt, k):
                    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
                    
                recon1 = reconstruct_svd(U1, S1, Vt1, k_recon)
                recon2 = reconstruct_svd(U2, S2, Vt2, k_recon)
                
                frob_err1 = np.linalg.norm(face_a_2d - recon1, ord="fro")
                frob_err2 = np.linalg.norm(face_b_2d - recon2, ord="fro")

                col_svd1, col_svd2 = st.columns(2)
                with col_svd1:
                    st.image(np.clip(recon1, 0.0, 1.0), caption=f"Rekonstruksi Foto A (k={k_recon})", use_container_width=True, clamp=True)
                    st.caption(f"Error Frobenius ($||A - A_k||_F$): {frob_err1:.4f}")
                with col_svd2:
                    st.image(np.clip(recon2, 0.0, 1.0), caption=f"Rekonstruksi Foto B (k={k_recon})", use_container_width=True, clamp=True)
                    st.caption(f"Error Frobenius ($||A - A_k||_F$): {frob_err2:.4f}")

                # Plot Top Singular Values
                fig_s, ax_s = plt.subplots(figsize=(6, 2.2))
                ax_s.plot(range(1, min(31, len(S1) + 1)), S1[:30], marker="o", label="Foto A", color="blue")
                ax_s.plot(range(1, min(31, len(S2) + 1)), S2[:30], marker="x", label="Foto B", color="red")
                ax_s.set_title("30 Singular Values Pertama (Representasi Energi SVD)")
                ax_s.set_xlabel("Rank (i)")
                ax_s.set_ylabel("Singular Value (\u03c3)")
                ax_s.grid(True, alpha=0.3)
                ax_s.legend()
                st.pyplot(fig_s, use_container_width=True)
                plt.close(fig_s)

                # 3. Vector Projections & 2D Eigenspace Scatter Plot
                st.markdown("#### 🎯 Proyeksi Vektor Wajah ke Eigenspace PCA")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    # Scatter plot PC1 vs PC2
                    fig_sc, ax_sc = plt.subplots(figsize=(4, 4))
                    use_aging_active = st.session_state.get("compare_use_aging", False)
                    z_a_orig = st.session_state.get("compare_z_a_orig")
                    
                    if use_aging_active and z_a_orig is not None:
                        w1_pix_orig = z_a_orig["pixel"].flatten()
                        w1_pix_aged = st.session_state.compare_z_a["pixel"].flatten()
                        
                        # Plot Foto A Original (biru muda/semi-transparan)
                        ax_sc.scatter(w1_pix_orig[0], w1_pix_orig[1], c="royalblue", alpha=0.5, s=100, label="Foto A (Asli)", zorder=3)
                        # Plot Foto A Aged (cyan/biru terang)
                        ax_sc.scatter(w1_pix_aged[0], w1_pix_aged[1], c="deepskyblue", s=100, label="Foto A (Aged)", zorder=3)
                        # Plot Foto B (merah)
                        ax_sc.scatter(w2_pix[0], w2_pix[1], c="red", s=100, label="Foto B (PC1, PC2)", zorder=3)
                        
                        # Panah pergeseran vektor penuaan (dari asli ke aged)
                        dx = w1_pix_aged[0] - w1_pix_orig[0]
                        dy = w1_pix_aged[1] - w1_pix_orig[1]
                        if np.hypot(dx, dy) > 1e-5:
                            ax_sc.arrow(w1_pix_orig[0], w1_pix_orig[1], dx, dy, 
                                        head_width=max(0.02, 0.05 * np.hypot(dx, dy)), 
                                        head_length=max(0.02, 0.05 * np.hypot(dx, dy)), 
                                        fc="cyan", ec="cyan", length_includes_head=True, 
                                        alpha=0.8, zorder=4, label="Vektor Penuaan")
                        
                        # Garis hubung dari Foto A ter-aged ke Foto B
                        ax_sc.plot([w1_pix_aged[0], w2_pix[0]], [w1_pix_aged[1], w2_pix[1]], "k--", alpha=0.5, zorder=2)
                        
                        ax_sc.plot([0, w1_pix_orig[0]], [0, w1_pix_orig[1]], "b-", alpha=0.15)
                        ax_sc.plot([0, w1_pix_aged[0]], [0, w1_pix_aged[1]], "b-", alpha=0.3)
                    else:
                        ax_sc.scatter(w1_pix[0], w1_pix[1], c="blue", s=100, label="Foto A (PC1, PC2)", zorder=3)
                        ax_sc.scatter(w2_pix[0], w2_pix[1], c="red", s=100, label="Foto B (PC1, PC2)", zorder=3)
                        ax_sc.plot([w1_pix[0], w2_pix[0]], [w1_pix[1], w2_pix[1]], "k--", alpha=0.5, zorder=2)
                        ax_sc.plot([0, w1_pix[0]], [0, w1_pix[1]], "b-", alpha=0.3)
                        
                    ax_sc.plot([0, w2_pix[0]], [0, w2_pix[1]], "r-", alpha=0.3)
                    ax_sc.scatter(0, 0, c="black", marker="x", s=50, label="Wajah Rata-rata")
                    ax_sc.set_xlabel("PC 1 Weight (w_1)")
                    ax_sc.set_ylabel("PC 2 Weight (w_2)")
                    ax_sc.grid(True, alpha=0.3)
                    ax_sc.legend()
                    st.pyplot(fig_sc, use_container_width=True)
                    plt.close(fig_sc)
                with col_p2:
                    # Bar plot PC1 to PC10
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    x_idx = np.arange(1, 11)
                    
                    if use_aging_active and z_a_orig is not None:
                        w1_pix_orig = z_a_orig["pixel"].flatten()
                        w1_pix_aged = st.session_state.compare_z_a["pixel"].flatten()
                        
                        width = 0.25
                        ax_bar.bar(x_idx - width, w1_pix_orig[:10], width, label="Foto A (Asli)", color="royalblue", alpha=0.6)
                        ax_bar.bar(x_idx, w1_pix_aged[:10], width, label="Foto A (Aged)", color="deepskyblue")
                        ax_bar.bar(x_idx + width, w2_pix[:10], width, label="Foto B", color="red")
                    else:
                        width = 0.35
                        ax_bar.bar(x_idx - width/2, w1_pix[:10], width, label="Foto A", color="blue")
                        ax_bar.bar(x_idx + width/2, w2_pix[:10], width, label="Foto B", color="red")
                        
                    ax_bar.set_xticks(x_idx)
                    ax_bar.set_xticklabels([f"PC {i}" for i in x_idx])
                    ax_bar.set_title("Perbandingan Bobot PC 1 - PC 10")
                    ax_bar.grid(True, alpha=0.3)
                    ax_bar.legend()
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close(fig_bar)

                # 4. Absolute Difference Heatmap
                st.markdown("#### 🗺️ Absolute Difference Heatmap")
                diff_img = np.abs(face_a_2d - face_b_2d)
                fig_hm, ax_hm = plt.subplots(figsize=(5, 3.5))
                cax = ax_hm.imshow(diff_img, cmap="hot")
                fig_hm.colorbar(cax, ax=ax_hm, fraction=0.046, pad=0.04)
                ax_hm.axis("off")
                ax_hm.set_title("Selisih Intensitas Spasial Absolut")
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)

            # --- SEKSI VALIDASI USER & NARASI ADAPTIF ---
            st.divider()
            st.markdown("### 💬 Validasi Hasil (Ground Truth)")
            st.caption("Bantu kami mengevaluasi kecocokan ini untuk mendapatkan analisis mendalam mengenai perilaku model.")

            ground_truth = st.radio(
                "Apakah kedua foto ini sebenarnya merupakan orang yang sama?",
                ["Pilih...", "Ya, mereka adalah orang yang sama", "Tidak, mereka adalah orang yang berbeda"],
                index=0,
                key="gt_compare_radio"
            )

            if ground_truth != "Pilih...":
                st.markdown("#### 📘 Analisis Adaptif Hasil Perbandingan")
                
                is_same = (ground_truth == "Ya, mereka adalah orang yang sama")
                cos_sim = result["cosine_similarity_eigenspace"]

                # Skenario 1: True Positive
                if is_similar and is_same:
                    st.success("**Kategori Hasil: True Positive (Verifikasi Sukses) ✅**")
                    st.markdown(f"""
                    Model **berhasil** mengenali bahwa kedua foto ini merepresentasikan individu yang sama.
                    * **Analisis Aljabar:** Sudut antara vektor proyeksi wajah A dan B di dalam ruang PCA berdimensi rendah sangat sempit (nilai Cosine Similarity `{cos_sim:.4f}` tinggi). Hal ini berarti fitur spasial global (seperti bentuk wajah, letak kontur pipi, dahi, dan rasio kepala) sangat konsisten.
                    * **Peran Preprocessing:** Proses standardisasi koordinat wajah dan histogram equalization berhasil meredam distorsi pencahayaan, sehingga model PCA fokus murni pada kesamaan fitur struktural wajah.
                    """)

                # Skenario 2: False Positive
                elif is_similar and not is_same:
                    st.warning("**Kategori Hasil: False Positive (Salah Tebak / Terkecoh) ⚠️**")
                    st.markdown(f"""
                    Model menyatakan kedua wajah ini **MIRIP**, namun sebenarnya mereka adalah **orang yang berbeda** (seperti pada kasus foto Presiden Jokowi dan Raffi Ahmad).

                    **Mengapa model Eigenfaces (PCA) bisa terkecoh dalam skenario ini?**
                    1. **Kesamaan Pose & Penyelarasan Piksel:** Kedua wajah diposisikan sangat simetris dan menghadap lurus ke depan (*frontal face*). Karena PCA sensitif terhadap keselarasan spasial pixel-ke-pixel, posisi wajah yang identik ini membuat proyeksi vektor mereka mengarah ke kuadran yang sama di ruang PCA.
                    2. **Dominasi Variansi Kasar (Pencahayaan & Siluet):** Komponen utama awal (seperti PC 1 hingga PC 3) menangkap variansi paling dominan dari data latih, yang biasanya merupakan **arah pencahayaan global** (arah datangnya cahaya) dan **siluet bentuk luar kepala**, bukan fitur identitas yang mendetail. Karena pencahayaan kedua foto ini mirip, model menganggap mereka mirip.
                    3. **Metode Bersifat Unsupervised:** PCA tidak dilatih secara terarah untuk membedakan orang (seperti halnya metode LDA/Fisherfaces atau Deep Learning embeddings). PCA hanya mereduksi dimensi untuk tujuan rekonstruksi visual, sehingga tidak peka terhadap detail mikro wajah (seperti bentuk mata, lipatan kelopak mata, atau lekukan cuping hidung) yang membedakan identitas asli manusia.
                    """)

                # Skenario 3: True Negative
                elif not is_similar and not is_same:
                    st.success("**Kategori Hasil: True Negative (Penolakan Tepat) ✅**")
                    st.markdown(f"""
                    Model **berhasil** mendeteksi perbedaan dan menyatakan kedua wajah ini berbeda orang.
                    * **Analisis Aljabar:** Vektor proyeksi wajah A dan wajah B mengarah ke sudut yang saling menjauh di dalam ruang eigen (Cosine Similarity `{cos_sim:.4f}` rendah atau negatif).
                    * **Analisis Visual:** Perbedaan struktural kasar (misalnya perbedaan lebar rahang, dahi, jarak antar mata, atau rasio aspek hidung) cukup dominan sehingga koordinat proyeksi mereka terpisah jauh di luar batas threshold keputusan `{st.session_state.threshold:.4f}`.
                    """)

                # Skenario 4: False Negative
                elif not is_similar and is_same:
                    st.error("**Kategori Hasil: False Negative (Gagal Mengenali) ❌**")
                    st.markdown(f"""
                    Model menyatakan kedua foto **TIDAK MIRIP**, padahal sebenarnya mereka adalah **orang yang sama**.

                    **Faktor-faktor yang menyebabkan kegagalan deteksi pada PCA:**
                    1. **Variasi Non-Linear yang Ekstrem:** Sebagai metode linear, PCA sangat rentan gagal jika ada perbedaan non-linear yang kontras seperti **perubahan usia** yang signifikan (kerutan, struktur lemak wajah), **ekspresi wajah** yang sangat berbeda (tertawa lebar vs datar), atau tumbuhnya **kumis/jenggot**.
                    2. **Kemiringan Pose (Rotasi Kepala):** Jika kepala sedikit menoleh atau miring, koordinat piksel mata, hidung, dan mulut akan bergeser secara spasial. PCA akan menganggap pergeseran ini sebagai perbedaan objek/identitas yang berbeda.
                    3. **Perubahan Pencahayaan Ekstrem:** Bayangan yang tajam pada salah satu foto wajah dapat mendistorsi nilai piksel secara drastis, mengalahkan kesamaan fitur geometris wajah itu sendiri.
                    """)

                # Penjelasan kontekstual nilai Cosine Similarity
                st.markdown("##### 📊 Analisis Geometris Nilai Cosine Similarity")
                if cos_sim > 0.7:
                    st.info(f"Nilai Cosine Similarity sebesar **{cos_sim:.4f}** tergolong **SANGAT TINGGI**. Secara geometris, sudut proyeksi kedua wajah sangat sempit (hampir searah). Ini menandakan siluet kepala, rasio kasar, dan pencahayaan spasial kedua foto sangat serupa.")
                elif 0.3 < cos_sim <= 0.7:
                    st.info(f"Nilai Cosine Similarity sebesar **{cos_sim:.4f}** tergolong **SEDANG**. Terdapat beberapa komponen umum yang serupa (seperti bentuk wajah oval atau pencahayaan), tetapi detail struktur spasial wajah mulai menunjukkan ketidaksesuaian.")
                else:
                    st.info(f"Nilai Cosine Similarity sebesar **{cos_sim:.4f}** tergolong **RENDAH/NEGATIF**. Sudut proyeksi kedua wajah hampir tegak lurus (ortogonal) atau berlawanan arah di ruang eigen, membuktikan kesamaan bentuk visual sangat minim.")


# ============================================================
# TAB IDENTIFY
# ============================================================
def render_identify_tab():
    if not model_is_ready():
        st.warning("Model belum dilatih. Buka tab Train terlebih dahulu.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        use_face_detection = st.checkbox(
            "Face detection", value=True, key="identify_face_detection",
        )
    with col2:
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)

    query_file = st.file_uploader("Foto query", type=["jpg", "jpeg", "png"],
                                   key="identify_query")

    if query_file is not None:
        with st.expander("🔍 Detail Langkah-langkah Preprocessing (Foto Query)", expanded=True):
            render_step_visualizer(query_file, use_face_detection, st.session_state.equalize_hist, title="Query")

        if st.button("Identifikasi Wajah", type="primary"):
            try:
                with st.spinner("Memproses..."):
                    vec, thumb = load_uploaded_image_as_array(
                        query_file, st.session_state.image_size,
                        use_face_detection,
                        equalize_hist=st.session_state.equalize_hist)
                    result = _recognize_from_vector(
                        vec,
                        st.session_state.model,
                        st.session_state.X_pca,
                        st.session_state.labels,
                        threshold=st.session_state.threshold,
                        top_k=top_k,
                    )
            except ValueError as e:
                st.error(str(e))
                return

            st.divider()
            col_q, col_r = st.columns([1, 2])
            with col_q:
                st.image(thumb, caption="Query Ter-preprocess", use_container_width=True)
            with col_r:
                if result.best_label == "Tidak dikenal":
                    st.warning(
                        f"### Tidak dikenal ❌\n"
                        f"Best similarity: {result.best_similarity:.4f} "
                        f"(di bawah threshold {result.threshold:.4f})"
                    )
                else:
                    st.success(
                        f"### {result.best_label} ✅\n"
                        f"Similarity: {result.best_similarity:.4f}  |  "
                        f"Confidence: {result.best_confidence_percent:.2f}%"
                    )

            st.markdown(f"**Top-{top_k} kandidat terdekat dari database**")

            labels_arr = st.session_state.labels
            X_db = st.session_state.X
            image_shape = st.session_state.image_shape

            n_display = min(top_k, 5)
            cols = st.columns(n_display)
            
            for i in range(n_display):
                idx = result.top_k_indices[i]
                sim = result.top_k_similarities[i]
                with cols[i]:
                    thumb_db = np.clip(X_db[idx].reshape(image_shape), 0.0, 1.0)
                    st.image(
                        thumb_db, 
                        caption=f"#{i+1}: {labels_arr[idx]} (sim={sim:.4f})", 
                        use_container_width=True
                    )

            table_data = {
                "Rank": list(range(1, top_k + 1)),
                "Label": [labels_arr[i] for i in result.top_k_indices],
                "Cosine Similarity": [f"{s:.4f}" for s in result.top_k_similarities],
                "Raw Confidence (%)": [
                    f"{max(0.0, s)*100:.2f}" for s in result.top_k_similarities
                ],
            }
            st.table(table_data)


# ============================================================
# TAB ABOUT (Penjelasan Mendalam Alur Matematika & Program)
# ============================================================
def render_about_tab():
    st.markdown(r"""
## Penjelasan Mendalam: Eigenfaces, Multimodal Fusion dan Simulasi Usia

Aplikasi ini mengimplementasikan metode pengenalan wajah klasik **Eigenfaces** (Turk & Pentland, 1991) yang telah dikembangkan secara signifikan menggunakan **Principal Component Analysis (PCA)** berbasis **Singular Value Decomposition (SVD)**, dipadukan dengan **Fusi Fitur Multimodal (Pixel + LBP + HOG)** dan **Simulasi Penuaan (Aging Vectors)**.

Di bawah ini adalah penjelasan mendalam alur matematis dan mekanisme pengolahan wajah di program ini.

---

### Alur Kerja dan Langkah-Langkah Matematis

```mermaid
graph TD
    A[Gambar Asli BGR] -->|1. Konversi Gray| B[Grayscale]
    B -->|2. LBF Alignment / Haar| C[Penyelarasan Wajah]
    C -->|3. Standardisasi| D[Resized W, H]
    D -->|4. CLAHE Enhancement| E[CLAHE Contrast]
    E -->|5. Ekstraksi Fitur| F[Pixel / LBP / HOG]
    F -->|6. Reduksi Dimensi| G[Subspace PCA Projections]
    G -->|7. Evaluasi Fusi| H[Sensor Fusion Metric]
```

#### 1. Konversi Saluran Warna (Grayscale ITU-R BT.601)
Gambar asli dikonversi menjadi gambar satu saluran (*grayscale*) menggunakan formula standar luminance untuk meniru persepsi mata manusia:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
Nilai kecerahan piksel dinormalisasi dari skala $[0, 255]$ ke $[0, 1]$ demi kestabilan numerik proses SVD.

#### 2. Penyelarasan Wajah 68-Landmark LBF (LBF Face Alignment)
Untuk memitigasi distorsi akibat kemiringan kepala, kita menggunakan model **Facemark LBF**. Model mendeteksi 68 titik koordinat wajah, menghitung kemiringan sudut mata kiri dan kanan, lalu memutar (*rotate*) serta menskalakan gambar agar jarak antar mata konstan berada pada 40% dari lebar total gambar. Terakhir, gambar ditranslasi agar pusat mata berada tepat di koordinat spasial $x=50\%$ dan $y=35\%$.

#### 3. Peningkatan Kontras Adaptif Lokal (CLAHE)
Untuk mengurangi dampak bayangan tajam tanpa merusak fitur geometris wajah, kita menerapkan **CLAHE (Contrast Limited Adaptive Histogram Equalization)** dengan ambang batas klip $1.5$ pada grid adaptif $8 \times 8$.

#### 4. Ruang Fitur Multimodal (Fusion Mode)
Guna meningkatkan ketangguhan pencocokan, sistem mengekstrak tiga modalitas fitur:
*   **Pixel Intensity (Pixel PCA):** Menangkap struktur wajah makro dan siluet kepala.
*   **Local Binary Patterns (LBP PCA):** Mengekstrak tekstur mikro kulit (sangat kebal terhadap perubahan cahaya global).
*   **Histogram of Oriented Gradients (HOG PCA):** Mengekstrak kontur arah gradien tepi wajah (sangat kebal terhadap kerutan/perubahan usia).

Ketiganya diproyeksikan ke sub-eigenspace masing-masing, dihitung nilai kemiripan sudut kosinusnya, lalu digabungkan secara tertimbang (*weighted fusion*).

#### 5. Pengurangan Komponen Awal (PC1-3 Removal) & Penalti Jarak
Tiga komponen utama pertama (PC 1, PC 2, PC 3) menangkap variansi paling dominan dari data latih, yang biasanya merupakan **arah pencahayaan global** dan bentuk siluet kasar kepala, bukan identitas wajah. Oleh karena itu, komponen-komponen awal ini dibuang dalam evaluasi jarak Euclidean untuk memperkuat pembedaan identitas asli wajah (*Identity Verification*). Jarak Euclidean ini kemudian menjadi faktor penalti bagi skor kemiripan akhir.

---

### Dataset Pelatihan Gabungan dan Aging Vector

Model Eigenspace pada berkas `pretrained_eigenspace.npz` dilatih menggunakan tiga dataset dengan peran fungsional yang berbeda:

1.  **Labeled Faces in the Wild (LFW):** Memperkaya keragaman variasi pose, ekspresi, pencahayaan, dan latar belakang global.
2.  **Indonesia Muslim Student Face Dataset (IMSFD):** Sebanyak 26.760 gambar dari 68 subjek mahasiswa Muslim Indonesia. Dataset ini melatih eigenspace agar peka terhadap struktur wajah **Asia Tenggara** serta oklusi penutup kepala seperti **hijab** dan **peci**.
3.  **FG-NET Aging Database:** Wajah multi-usia (1 s.d. 69 tahun) untuk melatih **Simulasi Penuaan**.

#### Mekanisme Simulasi Usia (Aging Vector)
Arah penuaan matematis dihitung di ruang PCA menggunakan formula selisih rata-rata proyeksi wajah dewasa ($\ge 18$ tahun) dikurangi rata-rata proyeksi wajah anak-anak ($\le 12$ tahun):
$$\vec{v}_{aging} = \vec{\mu}_{adult} - \vec{\mu}_{child}$$
Saat membandingkan dua wajah dengan perbedaan usia yang kontras (misal foto masa kecil vs masa kini), sistem memodulasi koordinat wajah anak di ruang eigen dengan menjumlahkan vektor penuaan:
$$\vec{z}_{aged} = \vec{z}_{child} + \alpha \cdot \vec{v}_{aging}$$
Ini mengompensasi pergeseran koordinat linear akibat penuaan biologis, sehingga pencocokan wajah lintas usia menjadi jauh lebih presisi.
""")


# ============================================================
# MAIN
# ============================================================
def main():
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["Train", "Compare", "Identify", "About"])

    with tab1:
        render_train_tab()
    with tab2:
        render_compare_tab()
    with tab3:
        render_identify_tab()
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
