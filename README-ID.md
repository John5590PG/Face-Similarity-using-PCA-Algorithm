# SISTEM KEMIRIPAN WAJAH: EIGENFACES VIA SVD (VERSI OPTIMASI FUSI DAN AGING)

Implementasi modern dan ringan dari sistem pengenalan wajah klasik **Eigenfaces** (Turk & Pentland, 1991). Sistem ini menggunakan **Principal Component Analysis (PCA)** melalui dekomposisi **Singular Value Decomposition (SVD)** untuk mereduksi dimensi gambar wajah berdimensi tinggi ke ruang bagian linear (*eigenspace*) yang kompak.

Pembaruan utama versi ini adalah integrasi fitur **Sensor Fusion (Pixel, LBP, HOG)** dan **Aging Vectors** untuk verifikasi wajah lintas usia, serta dioptimalkan menggunakan tiga dataset latih besar: **LFW**, **FG-NET**, dan **IMSFD (Wajah Asia Tenggara dan Hijab)**.

---

## Arsitektur Teknis dan Pembaruan Sistem

1.  **Pelatihan Eigenspace Gabungan (LFW + FG-NET + IMSFD):**
    - **LFW:** Memperkaya ruang wajah dengan beragam ekspresi dan pose umum.
    - **FG-NET:** Digunakan untuk melacak perubahan penuaan fisik wajah (usia 1–69 tahun).
    - **IMSFD (Indonesia Muslim Student Face Dataset):** Memperkaya pencocokan wajah dengan karakteristik lokal wajah Asia Tenggara dan variasi berhijab secara presisi.
2.  **Sensor Fusion (Multimodalitas):**
    - **Pixel PCA:** Menangkap bentuk siluet wajah kasar dan pencahayaan global.
    - **LBP PCA (Local Binary Patterns):** Menangkap tekstur lokal kulit wajah (tahan terhadap bayangan/cahaya ekstrem).
    - **HOG PCA (Histogram of Oriented Gradients):** Menangkap bentuk kontur geometris wajah yang stabil.
3.  **Kompensasi Usia (Aging Vector):**
    - Model memproyeksikan data FG-NET ke ruang PCA, membagi data anak-anak (≤12 tahun) dan dewasa (≥18 tahun), lalu menghitung selisih rata-ratanya sebagai arah/vektor penuaan (aging_vector).
    - Sistem dapat melakukan simulasi penuaan (aging) atau peremajaan (de-aging) pada vektor wajah sebelum membandingkannya.

---

## Struktur Repositori (folder improved)

*   app.py: Frontend UI Streamlit dan lapisan visualisasi tahapan preprocessing.
*   face_similarity.py: Mesin aljabar komputasi utama (ekstraksi fitur LBP, HOG, fusi kemiripan, dan dekomposisi).
*   pretrained_eigenspace.npz: Payload model terkompresi hasil training Kaggle (Pixel, LBP, HOG PCA, dan aging vectors).
*   lbfmodel.yaml: Parameter model landmark mata & wajah (LBF Facemark).
*   requirements.txt & requirements.md: Daftar pustaka dependensi dan petunjuk instalasi lingkungan Python.

---

## Instalasi dan Menjalankan Program

### 1. Pasang Dependensi
Pastikan Python 3.9 s.d. 3.11 sudah terpasang. Jalankan instalasi dependensi di direktori proyek:
```bash
pip install -r requirements.txt
```

### 2. Jalankan Streamlit
Untuk memulai dasbor server lokal Streamlit secara interaktif:
```bash
streamlit run app.py
```

---

## Cara Kerja Penggunaan Model Pra-Latih (.npz)

Saat aplikasi dijalankan:
1.  Buka tab **Train** di antarmuka Streamlit.
2.  Centang pilihan **Model Pra-Latih (.npz)**.
3.  Sistem akan otomatis mendeteksi berkas `pretrained_eigenspace.npz` di folder `improved/`.
4.  Klik **Latih sekarang**. Model instan termuat dengan variance explained, visualisasi Scree plot, dan siap digunakan pada tab **Compare** atau **Identify** dengan sensor fusi aktif secara otomatis.
