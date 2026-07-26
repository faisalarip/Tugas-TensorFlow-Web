# Aplikasi Prediksi Gambar CIFAR-10

Project ini adalah tugas sederhana klasifikasi gambar menggunakan Convolutional Neural Network (CNN) dengan dataset CIFAR-10. Aplikasi web dibuat menggunakan Flask agar user dapat mengunggah gambar dan melihat hasil prediksi kelas gambar.

## Isi Project

- `train_cnn.py` untuk training model CNN
- `app.py` untuk website Flask
- `utils/cifar10_utils.py` untuk preprocessing gambar
- `requirements.txt` untuk daftar dependensi

## Jika TensorFlow crash di Mac

Jika muncul error seperti `mutex lock failed: Invalid argument`, bersihkan dulu paket lama yang bentrok:

```bash
python3 -m pip uninstall -y tensorflow streamlit pyarrow tensorboard keras
python3 -m pip install -r requirements.txt
```

Lalu cek apakah TensorFlow sudah bisa di-import:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

## Langkah Menjalankan Project

### 1. Install dependensi

```bash
python3 -m pip install -r requirements.txt
```

### 2. Training model CNN

```bash
python3 train_cnn.py
```

Model akan disimpan di `models/cifar10_cnn.keras`.

Opsional, jika ingin mengatur jumlah epoch dan batch size:

```bash
python3 train_cnn.py --epochs 10 --batch-size 64
```

### 3. Jalankan website

```bash
python3 app.py
```

Setelah server berjalan, buka browser dan akses:

```text
http://127.0.0.1:5000
```

## Fitur Website

- Upload gambar
- Prediksi kelas gambar
- Tampilkan hasil prediksi

## Deploy ke Render

1. Push project ini ke GitHub
2. Buka Render dan pilih **New +** -> **Blueprint**
3. Hubungkan repository `Tugas-TensorFlow-Web`
4. Render akan membaca file `render.yaml` otomatis
5. Jalankan deploy dan tunggu sampai service aktif

## Kelas CIFAR-10

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck
