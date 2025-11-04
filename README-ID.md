# Virtual Try-On dengan Godot dan Python

Proyek ini adalah aplikasi *virtual try-on* secara *real-time* yang menggunakan *pipeline computer vision* klasik untuk deteksi wajah dan mengintegrasikannya dengan *game engine* Godot untuk *rendering*. *Backend* dibuat dengan Python dan OpenCV, sedangkan *frontend* adalah aplikasi Godot 4. Kedua bagian berkomunikasi melalui koneksi jaringan UDP.

## Fitur

- **Deteksi Wajah Real-time**: Menggunakan *pipeline Haar Cascades*, fitur *ORB*, *Bag of Visual Words (BoVW)*, dan *classifier Support Vector Machine (SVM)* untuk mendeteksi wajah dalam umpan webcam.
- **Overlay Topeng Virtual**: Menempatkan berbagai topeng bertema Halloween pada wajah yang terdeteksi.
- **Pemilihan Topeng Dinamis**: Memungkinkan pengguna untuk mengganti topeng secara *real-time* dari UI Godot.
- **Integrasi Godot**: Umpan video yang diproses dari *backend* Python di-*stream* ke aplikasi Godot untuk ditampilkan.
- **Komunikasi UDP**: Secara efisien melakukan *streaming* data video dan perintah antara *backend* Python dan *frontend* Godot.

## Arsitektur

Proyek ini dibagi menjadi dua komponen utama:

1.  **Backend (`backend/`)**: Aplikasi Python yang bertanggung jawab untuk:
    - Menangkap video dari webcam.
    - Melakukan deteksi wajah menggunakan model *SVM* yang dilatih khusus.
    - Menempatkan topeng yang dipilih pada wajah yang terdeteksi.
    - Meng-*encode* *frame* video akhir sebagai gambar JPEG.
    - Melakukan *streaming* data gambar ke klien Godot melalui *UDP*.
    - Mendengarkan perintah dari klien Godot (misalnya, untuk mengganti topeng).

2.  **Frontend (`godot/`)**: Aplikasi Godot 4 yang:
    - Menerima data gambar JPEG dari *backend* Python melalui *UDP*.
    - Meng-*decode* gambar dan menampilkannya secara *real-time*.
    - Menyediakan antarmuka pengguna untuk memilih topeng yang berbeda.
    - Mengirim perintah ke *backend* untuk mengubah topeng yang sedang ditampilkan.

### Protokol Komunikasi

-   **Video Stream (Python -> Godot)**: *Backend* mengirimkan objek JSON melalui *UDP* ke port `5555`. Setiap objek berisi `image` (sebagai JPEG yang di-*encode* base64), `frame_id`, `fps`, dan jumlah `faces` yang terdeteksi.
-   **Perintah (Godot -> Python)**: Klien Godot mengirimkan perintah JSON ke *backend* di port `5556`. Perintah utamanya adalah `change_mask`, yang menyertakan nama file dari topeng yang diinginkan.

## Cara Kerja

### Pipeline Backend

Proses deteksi wajah di *backend* mengikuti langkah-langkah berikut:

1.  **Proposal ROI**: *Classifier Haar Cascade* digunakan untuk mengusulkan *Regions of Interest (ROI)* di mana kemungkinan besar terdapat wajah.
2.  **Ekstraksi Fitur**: Untuk setiap ROI, *keypoints* dan *descriptor Oriented FAST and Rotated BRIEF (ORB)* diekstraksi.
3.  **Bag of Visual Words (BoVW)**: *Descriptor ORB* di-*encode* menjadi histogram menggunakan *"codebook" BoVW* yang sudah dilatih sebelumnya (model *KMeans*). Ini mengubah *descriptor* dengan panjang variabel menjadi vektor fitur berukuran tetap.
4.  **Klasifikasi SVM**: Vektor fitur dimasukkan ke dalam *classifier Support Vector Machine (SVM)* yang telah dilatih sebelumnya, yang menentukan apakah ROI berisi wajah.
5.  **Overlay Topeng**: Jika wajah terdeteksi, topeng yang dipilih akan ditempatkan di atas area wajah. Sistem juga dapat mendeteksi sudut mata untuk memutar topeng agar lebih pas.
6.  **Streaming UDP**: *Frame* akhir di-*encode* dan dikirim ke Godot.

### Frontend (Godot)

Aplikasi Godot relatif sederhana:

1.  **UDP Listener**: Sebuah `PacketPeerUDP` diatur untuk mendengarkan data yang masuk pada port yang ditentukan.
2.  **Pemrosesan Frame**: Ketika sebuah paket diterima, data JSON di-*parse*. *String* gambar base64 di-*decode* menjadi *buffer* JPEG.
3.  **Pembaruan Tekstur**: *Buffer* JPEG dimuat ke dalam `Image`, yang kemudian digunakan untuk membuat `ImageTexture`. Tekstur ini ditampilkan dalam *node* `TextureRect`, yang secara efektif menampilkan *stream* video.
4.  **Interaksi UI**: Adegan UI terpisah memungkinkan pengguna untuk mengklik ikon topeng yang berbeda. Mengklik topeng akan mengirimkan perintah *UDP* ke *backend* untuk mengganti topeng yang aktif.

## Cara Menjalankan

### 1. Backend

Pertama, Anda harus menginstal Python dan *library* yang diperlukan.

```bash
# Arahkan ke direktori backend
cd backend

# Instal dependensi
pip install -r requirements.txt

# Jalankan server UDP
python webcam_udp_server.py
```

*Backend* akan mulai menangkap webcam, melakukan deteksi wajah, dan melakukan *streaming* output ke `127.0.0.1:5555`.

### 2. Frontend

Anda memerlukan Godot 4 untuk menjalankan *frontend*.

1.  Buka Godot Engine.
2.  Klik "Impor" dan arahkan ke direktori `godot/` di proyek ini. Pilih file `project.godot`.
3.  Setelah proyek diimpor, buka.
4.  Jalankan adegan utama (`MainMenu.tscn`) dengan menekan F5.

Aplikasi akan terbuka, dan Anda akan melihat umpan video dari *backend* Python. Anda kemudian dapat menavigasi ke adegan "Try On" untuk melihat topeng virtual dan memilih yang berbeda dari menu sebelah kanan.
