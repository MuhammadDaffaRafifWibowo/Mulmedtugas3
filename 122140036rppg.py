"""
main.py - Sistem Real-time Remote Photoplethysmography (rPPG)
Implementasi dengan metode POS (Plane-Orthogonal-to-Skin) untuk ketahanan terhadap gerakan
"""

import cv2
import numpy as np
from scipy import signal, fft
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp

class RealTimeRPPG:
    def __init__(self, camera_index=0, window_duration=10, fps=30):
        """
        Inisialisasi sistem rPPG real-time
        
        Parameters:
        -----------
        camera_index : int
            Indeks kamera (0 untuk webcam default)
        window_duration : int
            Durasi window analisis dalam detik
        fps : int
            Frame rate kamera
        """
        # Parameter sistem
        self.camera_index = camera_index
        self.fps = fps
        self.window_size = window_duration * fps  # Jumlah frame dalam window
        self.buffer_size = self.window_size * 2  # Buffer untuk sinyal
        
        # Inisialisasi buffer sinyal
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.bpm_buffer = deque(maxlen=30)  # Buffer untuk smoothing BPM
        
        # Parameter filter
        self.lowcut = 0.67  # Hz (40 BPM)
        self.highcut = 4.0   # Hz (240 BPM)
        self.nyquist = self.fps / 2
        self.low = self.lowcut / self.nyquist
        self.high = self.highcut / self.nyquist
        
        # Buat bandpass filter
        self.b, self.a = signal.butter(3, [self.low, self.high], btype='band')
        
        # Inisialisasi MediaPipe untuk deteksi wajah
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # model selection (0 untuk jarak dekat, 1 untuk jarak jauh)
            min_detection_confidence=0.5
        )
        
        # Variabel untuk tracking ROI
        self.roi_points = None
        self.roi_stability_counter = 0
        self.roi_stability_threshold = 5
        
        # Variabel untuk visualisasi
        self.fig, self.axes = None, None
        self.animation = None
        self.setup_visualization()
        
        # Statistik
        self.frame_count = 0
        self.processing_times = []
        self.current_bpm = 0
        
    def setup_visualization(self):
        """Setup plot untuk visualisasi real-time"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Subplot 1: Video feed dengan ROI
        self.video_ax = self.axes[0, 0]
        self.video_ax.set_title("Video Input with ROI")
        self.video_ax.axis('off')
        self.video_img = self.video_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Subplot 2: Sinyal rPPG yang difilter
        self.signal_ax = self.axes[0, 1]
        self.signal_ax.set_title("Filtered rPPG Signal")
        self.signal_ax.set_xlabel("Time (s)")
        self.signal_ax.set_ylabel("Amplitude")
        self.signal_line, = self.signal_ax.plot([], [], 'b-', linewidth=2)
        self.signal_ax.grid(True)
        
        # Subplot 3: Spektrum frekuensi
        self.spectrum_ax = self.axes[1, 0]
        self.spectrum_ax.set_title("Frequency Spectrum")
        self.spectrum_ax.set_xlabel("Frequency (Hz)")
        self.spectrum_ax.set_ylabel("Magnitude")
        self.spectrum_ax.set_xlim(0, 4)
        self.spectrum_line, = self.spectrum_ax.plot([], [], 'r-', linewidth=2)
        self.spectrum_ax.grid(True)
        
        # Subplot 4: BPM over time
        self.bpm_ax = self.axes[1, 1]
        self.bpm_ax.set_title("Heart Rate (BPM)")
        self.bpm_ax.set_xlabel("Time (s)")
        self.bpm_ax.set_ylabel("BPM")
        self.bpm_line, = self.bpm_ax.plot([], [], 'g-', linewidth=2, marker='o')
        self.bpm_ax.grid(True)
        self.bpm_ax.set_ylim(40, 180)  # Rentang normal BPM
        
    def detect_face_roi(self, frame):
        """
        Mendeteksi wajah dan menentukan ROI menggunakan MediaPipe
        
        Parameters:
        -----------
        frame : numpy array
            Frame video input
            
        Returns:
        --------
        roi : numpy array or None
            ROI yang berisi area kulit wajah
        roi_rect : tuple or None
            Koordinat ROI (x, y, w, h)
        """
        # Konversi BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]  # Ambil wajah pertama
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Perbaiki koordinat jika di luar batas
            x, y = max(0, x), max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)
            
            # Pilih area pipi sebagai ROI (lebih stabil)
            roi_height = int(height * 0.3)
            roi_y = y + int(height * 0.4)
            
            # Pastikan ROI tidak terlalu kecil
            if width > 50 and roi_height > 30:
                roi_rect = (x, roi_y, width, roi_height)
                roi = frame[roi_y:roi_y+roi_height, x:x+width]
                
                # Update ROI stability counter
                if self.roi_points is not None:
                    dx = abs(x - self.roi_points[0])
                    dy = abs(roi_y - self.roi_points[1])
                    if dx < 10 and dy < 10:
                        self.roi_stability_counter += 1
                    else:
                        self.roi_stability_counter = 0
                
                self.roi_points = (x, roi_y, width, roi_height)
                return roi, roi_rect
        
        return None, None
    
    def extract_signal_pos(self, roi):
        """
        Ekstraksi sinyal rPPG menggunakan metode POS (Plane-Orthogonal-to-Skin)
        
        Parameters:
        -----------
        roi : numpy array
            Region of Interest
            
        Returns:
        --------
        pulse_signal : float
            Nilai sinyal rPPG pada frame ini
        """
        # Metode POS: Projection onto Skin-tone plane
        # Langkah 1: Normalisasi kanal warna
        R = roi[:, :, 2].mean()
        G = roi[:, :, 1].mean()
        B = roi[:, :, 0].mean()
        
        # Hindari pembagian dengan nol
        if G == 0:
            return 0
        
        # Langkah 2: Hitung chrominance signals
        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B
        
        # Langkah 3: Alpha tuning (biasanya ~0.5 untuk kulit)
        alpha = 0.5
        S = Xs - alpha * Ys
        
        # Langkah 4: Standardization
        if np.std(S) > 0:
            S = (S - np.mean(S)) / np.std(S)
        
        return S
    
    def process_signal(self):
        """
        Memproses sinyal rPPG dalam buffer
        
        Returns:
        --------
        bpm : float
            Estimated heart rate in BPM
        filtered_signal : numpy array
            Sinyal yang sudah difilter
        """
        if len(self.signal_buffer) < self.window_size:
            return 0, np.array([])
        
        # Ambil data dari buffer
        signal_data = np.array(list(self.signal_buffer))
        time_data = np.array(list(self.time_buffer))
        
        # Potong data sesuai window size
        signal_window = signal_data[-self.window_size:]
        time_window = time_data[-self.window_size:]
        
        # 1. Detrending menggunakan moving average
        window_size = min(30, len(signal_window))
        if window_size % 2 == 0:
            window_size -= 1
        
        ma = signal.medfilt(signal_window, kernel_size=window_size)
        detrended = signal_window - ma
        
        # 2. Normalisasi
        if np.std(detrended) > 0:
            normalized = (detrended - np.mean(detrended)) / np.std(detrended)
        else:
            normalized = detrended
        
        # 3. Bandpass filtering
        filtered = signal.filtfilt(self.b, self.a, normalized)
        
        # 4. Hitung BPM menggunakan FFT
        n = len(filtered)
        if n > 0:
            # FFT
            fft_values = fft.fft(filtered)
            frequencies = fft.fftfreq(n, d=1/self.fps)
            
            # Ambil hanya frekuensi positif dalam rentang yang diinginkan
            mask = (frequencies >= self.lowcut) & (frequencies <= self.highcut)
            positive_freq = frequencies[mask]
            positive_fft = np.abs(fft_values[mask])
            
            if len(positive_freq) > 0:
                # Cari frekuensi dominan
                dominant_idx = np.argmax(positive_fft)
                dominant_freq = positive_freq[dominant_idx]
                
                # Konversi ke BPM
                bpm = dominant_freq * 60
                
                # Validasi BPM (rentang fisiologis normal)
                if 40 <= bpm <= 180:
                    # Smoothing dengan moving average
                    self.bpm_buffer.append(bpm)
                    smoothed_bpm = np.mean(self.bpm_buffer)
                    self.current_bpm = smoothed_bpm
                    
                    return smoothed_bpm, filtered
        
        return 0, np.array([])
    
    def draw_overlay(self, frame, roi_rect, bpm):
        """
        Gambar overlay informasi pada frame
        
        Parameters:
        -----------
        frame : numpy array
            Frame asli
        roi_rect : tuple
            Koordinat ROI (x, y, w, h)
        bpm : float
            Estimated BPM
        """
        # Gambar bounding box ROI
        if roi_rect:
            x, y, w, h = roi_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Tambahkan teks ROI
            cv2.putText(frame, "ROI", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Tambahkan informasi BPM
        bpm_text = f"Heart Rate: {bpm:.1f} BPM"
        cv2.putText(frame, bpm_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Tambahkan frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status stabilitas
        stability = min(self.roi_stability_counter / self.roi_stability_threshold, 1.0)
        cv2.rectangle(frame, (20, 100), (220, 120), (100, 100, 100), 1)
        cv2.rectangle(frame, (20, 100), (20 + int(200 * stability), 120), 
                     (0, int(255 * stability), 0), -1)
        cv2.putText(frame, "Stability", (230, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def update_plots(self, frame, filtered_signal, bpm):
        """
        Update plot visualisasi
        
        Parameters:
        -----------
        frame : numpy array
            Frame video
        filtered_signal : numpy array
            Sinyal yang sudah difilter
        bpm : float
            Estimated BPM
        """
        # Update video feed
        self.video_img.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Update sinyal plot
        if len(filtered_signal) > 0:
            time_axis = np.linspace(-len(filtered_signal)/self.fps, 0, len(filtered_signal))
            self.signal_line.set_data(time_axis, filtered_signal)
            self.signal_ax.relim()
            self.signal_ax.autoscale_view()
            
            # Update spectrum plot
            n = len(filtered_signal)
            if n > 0:
                fft_vals = np.abs(fft.fft(filtered_signal))
                freqs = fft.fftfreq(n, d=1/self.fps)
                mask = (freqs >= 0) & (freqs <= 4)
                
                self.spectrum_line.set_data(freqs[mask], fft_vals[mask])
                self.spectrum_ax.relim()
                self.spectrum_ax.autoscale_view()
        
        # Update BPM plot
        self.bpm_buffer.append(bpm)
        bpm_values = list(self.bpm_buffer)
        bpm_time = np.arange(-len(bpm_values)+1, 1) * (1/self.fps)
        
        self.bpm_line.set_data(bpm_time, bpm_values)
        self.bpm_ax.relim()
        self.bpm_ax.autoscale_view()
        
        # Update judul dengan BPM terkini
        self.bpm_ax.set_title(f"Heart Rate: {bpm:.1f} BPM")
        
    def run(self):
        """
        Jalankan sistem rPPG real-time utama
        """
        # Buka kamera
        cap = cv2.VideoCapture(self.camera_index)
        
        # Set resolusi kamera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print("Sistem rPPG Real-time dimulai...")
        print("Tekan 'q' untuk keluar")
        print("Tekan 'r' untuk reset buffer")
        
        # Timer untuk FPS calculation
        start_time = time.time()
        
        try:
            while True:
                # Baca frame dari kamera
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame dari kamera")
                    break
                
                # Hitung processing time
                frame_start = time.time()
                
                # Deteksi wajah dan ekstraksi ROI
                roi, roi_rect = self.detect_face_roi(frame)
                
                if roi is not None:
                    # Ekstraksi sinyal menggunakan metode POS
                    pulse_signal = self.extract_signal_pos(roi)
                    
                    # Tambahkan ke buffer
                    current_time = time.time() - start_time
                    self.signal_buffer.append(pulse_signal)
                    self.time_buffer.append(current_time)
                    
                    # Proses sinyal jika buffer cukup
                    bpm, filtered_signal = self.process_signal()
                else:
                    bpm = 0
                    filtered_signal = np.array([])
                
                # Gambar overlay pada frame
                frame = self.draw_overlay(frame, roi_rect, bpm)
                
                # Update visualisasi plot
                self.update_plots(frame, filtered_signal, bpm)
                
                # Tampilkan frame
                cv2.imshow("Real-time rPPG System", frame)
                
                # Update frame counter
                self.frame_count += 1
                
                # Hitung FPS
                processing_time = time.time() - frame_start
                self.processing_times.append(processing_time)
                
                # Tampilkan FPS periodik
                if self.frame_count % 30 == 0:
                    avg_fps = 1 / np.mean(self.processing_times[-30:])
                    print(f"FPS: {avg_fps:.1f}, BPM: {bpm:.1f}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffer
                    self.signal_buffer.clear()
                    self.time_buffer.clear()
                    self.bpm_buffer.clear()
                    print("Buffer telah direset")
                
                # Update plot
                plt.pause(0.001)
        
        except KeyboardInterrupt:
            print("\nDiinterupsi oleh pengguna")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            plt.close()
            
            # Print statistics
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                print(f"\nStatistik:")
                print(f"Total frame: {self.frame_count}")
                print(f"Rata-rata processing time per frame: {avg_time*1000:.1f} ms")
                print(f"Rata-rata FPS: {1/avg_time:.1f}")
                if self.current_bpm > 0:
                    print(f"BPM akhir: {self.current_bpm:.1f}")

def main():
    """Fungsi utama untuk menjalankan sistem rPPG"""
    # Inisialisasi sistem
    rppg_system = RealTimeRPPG(
        camera_index=0,      # Gunakan webcam default
        window_duration=10,  # 10 detik window analisis
        fps=30              # Frame rate target
    )
    
    # Jalankan sistem
    rppg_system.run()

if __name__ == "__main__":
    main()

    """
=============================================
CATATAN PERBEDAAN DENGAN DEMO DI KELAS
=============================================

VISUALISASI LEBIH KOMPREHENSIF
   Demo kelas: Hanya tampilan wajah dengan bounding box
   Implementasi ini: 4 subplot terpisah:
   - Video input dengan ROI
   - Sinyal rPPG yang sudah difilter
   - Spektrum frekuensi (FFT)
   - Grafik BPM over time