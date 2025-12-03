# Real-time rPPG System for Vital Sign Monitoring

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/[your-username]/[your-repo-name]?style=for-the-badge&color=yellow)](https://github.com/[your-username]/[your-repo-name]/stargazers)

##  Deskripsi Proyek

Proyek ini mengimplementasikan sistem **rPPG (remote Photoplethysmography)** secara *real-time* untuk mendeteksi detak jantung (BPM - Beats Per Minute) seseorang hanya melalui video wajah menggunakan *webcam*.

Demonstrasi ini dirancang khusus untuk memvisualisasikan seluruh proses analisis sinyal rPPG, mulai dari input video hingga hasil detak jantung yang stabil, melalui tata letak (layout) yang informatif dan komprehensif.

## Fitur Demonstrasi

Implementasi ini mencakup dua mode tampilan utama: Mode Sederhana untuk deteksi cepat, dan Mode Analisis Penuh (Empat Subplot) untuk presentasi kelas.

### Mode Tampilan Penuh disaat dikelas, bedanya dengan saya, dibawah.

Antarmuka utama didesain sebagai layar penuh yang terbagi menjadi 4 subplot, memberikan wawasan mendalam tentang proses rPPG:

1.  **Video Input dengan ROI:** Menampilkan *feed* video dari *webcam* dengan *bounding box* di sekitar wajah dan menandai **Area of Interest (ROI)** yang digunakan untuk ekstraksi sinyal warna.
2.  **Sinyal rPPG yang Difilter:** Grafik real-time dari sinyal rPPG (perubahan intensitas warna) setelah melalui filter (seperti filter Butterworth atau IIR) untuk menghilangkan *noise* dan pergerakan.
3.  **Spektrum Frekuensi (FFT):** Grafik hasil analisis **Fast Fourier Transform (FFT)** dari sinyal rPPG, yang menunjukkan frekuensi dominan (tempat detak jantung berada). Puncak tertinggi dalam rentang BPM manusia akan diidentifikasi sebagai detak jantung saat ini.
4.  **Trend BPM:** Grafik garis yang menampilkan riwayat perubahan nilai BPM dari waktu ke waktu, memberikan gambaran stabilitas dan *trend* detak jantung pengguna.

### Mode Tampilan Sederhana

Mode ini menampilkan satu layar penuh dengan fokus pada *bounding box* wajah dan nilai BPM hasil akhir saja.

## 🛠️ Persyaratan Sistem & Instalasi

Proyek ini membutuhkan Python 3.8 ke atas (direkomendasikan Python 3.10+ untuk kompatibilitas `mediapipe`).

### 1. Klon Repositori

```bash
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]
https://gemini.google.com/share/c05ddd18d52e
