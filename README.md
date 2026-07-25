# Aplikasi Prediksi Gambar CIFAR-10

Project ini berisi website sederhana untuk tugas klasifikasi gambar CIFAR-10 menggunakan CNN dan Flask.

## Jika TensorFlow crash di Mac

Jika muncul error seperti `mutex lock failed: Invalid argument`, bersihkan dulu paket lama yang bentrok:

```bash
python3 -m pip uninstall -y tensorflow streamlit pyarrow tensorboard keras
python3 -m pip install -r requirements.txt
```

Lalu cek TensorFlow:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

## Install dependensi

```bash
python3 -m pip install -r requirements.txt
```

## Latih model CNN

```bash
python3 train_cnn.py
```

Model akan disimpan di `models/cifar10_cnn.keras`.

Opsional:

```bash
python3 train_cnn.py --epochs 10 --batch-size 64
```

## Jalankan website

```bash
python3 app.py
```

## Fitur

- Upload gambar
- Prediksi kelas gambar
- Tampilkan hasil prediksi
