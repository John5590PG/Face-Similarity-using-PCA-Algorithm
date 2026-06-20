# Dokumentasi Lingkungan Eksekusi dan Dependensi (Requirements)

Dokumen ini mendeskripsikan spesifikasi lingkungan kerja Python dan pustaka yang dibutuhkan untuk menjalankan aplikasi **Face Similarity** versi optimal secara lokal maupun mendeploynya ke **Streamlit Community Cloud**.

---

## Daftar Pustaka (Dependencies)

Pustaka-pustaka berikut wajib terpasang dan dicatat dalam `requirements.txt` untuk memastikan seluruh fitur aplikasi berjalan:

| Nama Pustaka | Peran dalam Aplikasi | Versi Rekomendasi |
|---|---|---|
| `streamlit` | Kerangka kerja *frontend* interaktif | `~=1.30.0` |
| `opencv-python-headless` | Pengolahan gambar, filter grayscale, resizing, CLAHE, Haar Cascade, dan LBF alignment. (Disarankan versi `headless` untuk deploy cloud agar tidak bentrok dengan pustaka sistem X11) | `~=4.9.0` |
| `numpy` | Operasi aljabar linear, matriks piksel, SVD, dan perhitungan spasial | `~=1.24.0` |
| `scikit-learn` | Pemodelan `PCA` (*Principal Component Analysis*) | `~=1.3.0` |
| `scikit-image` | Ekstraksi fitur tekstur wajah (`LBP`) dan fitur geometri wajah (`HOG`) | `~=0.22.0` |
| `matplotlib` | Visualisasi visual 20 komponen *eigenfaces* dan grafik Scree plot | `~=3.8.0` |
| `Pillow` | Manipulasi dan pembacaan berkas gambar (*image formatting*) | `~=10.0.0` |
| `pandas` | Pengolahan tabel skor kemiripan (ranking kandidat wajah) | `~=2.1.0` |

---

## Instalasi Lokal

### 1. Persiapan Environment
Disarankan untuk membuat virtual environment (venv) agar tidak bentrok dengan pustaka global Anda:
```bash
# Membuat venv baru
python -m venv venv

# Mengaktifkan venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Mengaktifkan venv (Linux/macOS)
source venv/bin/activate
```

### 2. Memasang Dependensi
Pasang seluruh pustaka di atas menggunakan berkas `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Panduan Deploy ke Streamlit Cloud

Saat mendeploy ke **Streamlit Community Cloud** via GitHub:
1.  Pastikan repositori GitHub Anda memiliki struktur folder `improved/` yang bersih.
2.  Pastikan berkas `requirements.txt` berada di tempat yang sama dengan berkas utama `app.py`.
3.  Di dashboard Streamlit Cloud, pilih **Main file path** ke: `improved/app.py`.
4.  Streamlit Cloud akan mendeteksi `requirements.txt` secara otomatis di direktori tersebut dan memasang semua pustaka yang tertulis.
