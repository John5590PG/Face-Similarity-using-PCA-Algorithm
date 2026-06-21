import json
import os

workspace_dir = os.path.dirname(os.path.abspath(__file__))
face_similarity_path = os.path.join(workspace_dir, "face_similarity.py")
app_path = os.path.join(workspace_dir, "app.py")
output_notebook_path = os.path.join(workspace_dir, "Face_Similarity_Colab.ipynb")

print("Membaca file sumber...")

# Membaca face_similarity.py
if not os.path.exists(face_similarity_path):
    raise FileNotFoundError("face_similarity.py tidak ditemukan di folder improved!")
with open(face_similarity_path, "r", encoding="utf-8") as f:
    face_similarity_content = f.read()

# Membaca app.py
if not os.path.exists(app_path):
    raise FileNotFoundError("app.py tidak ditemukan di folder improved!")
with open(app_path, "r", encoding="utf-8") as f:
    app_content = f.read()

# Membuat struktur sel notebook
cells = []

# Sel 1: Markdown Deskripsi
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Sistem Perbandingan Wajah Eigenfaces (PCA + LBP + HOG)\n",
        "Notebook ini dirancang untuk menjalankan aplikasi perbandingan wajah langsung di Google Colab.\n",
        "Anda dapat menjalankan antarmuka visual (Streamlit) atau membandingkan wajah secara terprogram melalui sel Python."
    ]
})

# Sel 2: Code Instalasi Dependensi
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Menginstal dependensi yang diperlukan\n",
        "!pip install streamlit opencv-contrib-python-headless scikit-learn scikit-image matplotlib Pillow pandas --quiet\n",
        "# Unduh cloudflared untuk alternatif tunnel yang lebih stabil dan bebas error dynamic module\n",
        "!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -q\n",
        "!chmod +x cloudflared-linux-amd64\n",
        "# Menginstal localtunnel sebagai opsi cadangan\n",
        "!npm install -g localtunnel --quiet\n",
        "print(\"✅ Instalasi dependensi selesai!\")"
    ]
})

# Sel 3: Code Upload Model & Download LBF Model
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 2. Unggah file model 'pretrained_eigenspace.npz'\n",
        "import os\n",
        "import urllib.request\n",
        "from google.colab import files\n",
        "\n",
        "if not os.path.exists(\"pretrained_eigenspace.npz\"):\n",
        "    print(\"Silakan unggah file 'pretrained_eigenspace.npz' Anda:\")\n",
        "    uploaded = files.upload()\n",
        "else:\n",
        "    print(\"✅ 'pretrained_eigenspace.npz' sudah ada di workspace.\")\n",
        "\n",
        "# Download lbfmodel.yaml secara otomatis agar tidak perlu mengunggah file 56MB secara manual\n",
        "LBF_MODEL_PATH = \"lbfmodel.yaml\"\n",
        "if not os.path.exists(LBF_MODEL_PATH):\n",
        "    print(\"⏳ Mengunduh lbfmodel.yaml (~53MB)...(Bisa memakan waktu 1-2 menit)\")\n",
        "    urllib.request.urlretrieve(\n",
        "        \"https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml\",\n",
        "        LBF_MODEL_PATH\n",
        "    )\n",
        "    print(\"✅ lbfmodel.yaml berhasil diunduh!\")\n",
        "else:\n",
        "    print(\"✅ lbfmodel.yaml sudah ada di workspace.\")"
    ]
})

# Sel 4: Code Menulis face_similarity.py
face_similarity_lines = ["%%writefile face_similarity.py\n"] + [line + "\n" for line in face_similarity_content.splitlines()]
if face_similarity_lines[-1] == "\n":
    face_similarity_lines.pop()

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": face_similarity_lines
})

# Sel 5: Code Menulis app.py
app_lines = ["%%writefile app.py\n"] + [line + "\n" for line in app_content.splitlines()]
if app_lines[-1] == "\n":
    app_lines.pop()

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": app_lines
})

# Sel 6: Markdown Cara Jalankan Streamlit Opsi A (Cloudflare)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Menjalankan Aplikasi Streamlit\n",
        "Terdapat dua opsi untuk mengekspos aplikasi Streamlit Anda ke URL publik. Opsi A menggunakan Cloudflare Tunnel sangat disarankan karena jauh lebih stabil dan tidak memiliki masalah pembatasan dynamic module (Slider/FileUploader).\n",
        "\n",
        "### OPSI A: Menggunakan Cloudflare Tunnel (Sangat Direkomendasikan)\n",
        "1. Jalankan sel Opsi A di bawah ini.\n",
        "2. Cari tautan berakhiran `.trycloudflare.com` di bagian log output (cari teks seperti `https://xxx.trycloudflare.com`).\n",
        "3. Klik tautan tersebut untuk membuka aplikasi Anda tanpa password."
    ]
})

# Sel 7: Code Jalankan Streamlit & Cloudflare Tunnel
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Menjalankan Streamlit di latar belakang pada port 8501\n",
        "import subprocess\n",
        "import time\n",
        "subprocess.Popen([\"streamlit\", \"run\", \"app.py\", \"--server.port\", \"8501\"])\n",
        "time.sleep(3)\n",
        "\n",
        "# Menghubungkan Cloudflare Tunnel ke port 8501\n",
        "!./cloudflared-linux-amd64 tunnel --url http://localhost:8501"
    ]
})

# Sel 8: Markdown Cara Jalankan Streamlit Opsi B (LocalTunnel)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### OPSI B: Menggunakan LocalTunnel (Alternatif)\n",
        "Jika Anda memilih opsi ini dan menemui error *TypeError: Failed to fetch dynamically imported module* (misalnya pada Slider atau FileUploader):\n",
        "* Harap lakukan **Hard Refresh** (`Ctrl + F5` atau `Ctrl + Shift + R`) pada browser Anda setelah melewati halaman password localtunnel.\n",
        "* Pastikan Anda **tidak** menggunakan mode Private/Incognito di browser agar cookies bypass localtunnel tidak diblokir.\n",
        "\n",
        "Cara menjalankan:\n",
        "1. Jalankan sel Opsi B di bawah ini.\n",
        "2. Salin alamat IP publik yang muncul pada baris pertama (misalnya: `34.125.43.120`).\n",
        "3. Klik tautan localtunnel (berakhiran `.loca.lt`).\n",
        "4. Tempel alamat IP tadi pada kotak **Tunnel Password** lalu klik submit."
    ]
})

# Sel 9: Code Jalankan Streamlit & Localtunnel
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Mendapatkan IP publik Colab untuk password localtunnel\n",
        "import urllib.request\n",
        "ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()\n",
        "print(f\"PASSWORD TUNNEL ANDA (IP Publik Colab): {ip}\")\n",
        "\n",
        "# Menjalankan Streamlit di latar belakang dan menghubungkan ke localtunnel\n",
        "import subprocess\n",
        "subprocess.Popen([\"streamlit\", \"run\", \"app.py\", \"--server.port\", \"8501\"])\n",
        "!npx localtunnel --port 8501"
    ]
})

# Sel 8: Markdown Penggunaan Programmatic
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Penggunaan Secara Programmatic (Tanpa Streamlit)\n",
        "Jika Anda hanya ingin membandingkan dua gambar secara terprogram di dalam notebook ini tanpa membuka web app, gunakan sel di bawah ini."
    ]
})

# Sel 10: Code Programmatic
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Unggah dua foto yang ingin dibandingkan\n",
        "from google.colab import files\n",
        "print(\"Unggah foto pertama (Wajah A):\")\n",
        "uploaded_a = files.upload()\n",
        "file_a = list(uploaded_a.keys())[0]\n",
        "\n",
        "print(\"Unggah foto kedua (Wajah B):\")\n",
        "uploaded_b = files.upload()\n",
        "file_b = list(uploaded_b.keys())[0]\n",
        "\n",
        "# Import modul face_similarity\n",
        "import face_similarity\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Muat model eigenspace\n",
        "model = face_similarity.load_pretrained_eigenspace_model(\"pretrained_eigenspace.npz\")\n",
        "if model is None:\n",
        "    raise FileNotFoundError(\"Gagal memuat model. Pastikan file 'pretrained_eigenspace.npz' sudah diunggah.\")\n",
        "\n",
        "# Jalankan langkah-langkah preprocessing untuk wajah A dan B\n",
        "steps_a = face_similarity.get_preprocessing_steps(file_a, image_size=model.image_shape, use_face_detection=True, equalize_hist=True)\n",
        "steps_b = face_similarity.get_preprocessing_steps(file_b, image_size=model.image_shape, use_face_detection=True, equalize_hist=True)\n",
        "\n",
        "vector_a = steps_a[\"7_flattened\"]\n",
        "vector_b = steps_b[\"7_flattened\"]\n",
        "\n",
        "# Jalankan perbandingan wajah\n",
        "result = face_similarity.compare_two_faces(vector_a, vector_b, model=model, threshold=0.80)\n",
        "\n",
        "print(f\"\\n=====================================\")\n",
        "print(f\"Hasil Perbandingan:\")\n",
        "print(f\"Kemiripan Cosine: {result.cosine_similarity:.4f}\")\n",
        "print(f\"Persentase Kemiripan: {result.confidence_percent:.2f}%\")\n",
        "print(f\"Status: {'Wajah Sama (Cocok)' if result.is_similar else 'Wajah Berbeda (Tidak Cocok)'}\")\n",
        "print(f\"Threshold: {result.threshold * 100:.2f}%\")\n",
        "print(f\"=====================================\\n\")\n",
        "\n",
        "# Visualisasi wajah yang disejajarkan\n",
        "aligned_a = steps_a[\"6_normalized\"]\n",
        "aligned_b = steps_b[\"6_normalized\"]\n",
        "\n",
        "if aligned_a is not None and aligned_b is not None:\n",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))\n",
        "    ax1.imshow(aligned_a, cmap=\"gray\")\n",
        "    ax1.set_title(\"Wajah A (Aligned & Normalized)\")\n",
        "    ax1.axis(\"off\")\n",
        "    \n",
        "    ax2.imshow(aligned_b, cmap=\"gray\")\n",
        "    ax2.set_title(\"Wajah B (Aligned & Normalized)\")\n",
        "    ax2.axis(\"off\")\n",
        "    plt.show()"
    ]
})

# Menyimpan notebook .ipynb
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(output_notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Berhasil menghasilkan notebook di: {output_notebook_path}")
