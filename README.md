# Video_Steganography_And_Video_Extraction_Using_Transform_Domain

## 📌 Overview
This project implements a highly secure video steganography system that hides a "secret" video completely inside a "cover" video. It uses **Alpha Blending** in the spatial/transform domain to embed the frames of the secret video into the cover video seamlessly. 

To provide an impenetrable layer of security, the secret video is encrypted using **AES-256 (Advanced Encryption Standard in GCM mode)** before any steganography occurs. The extraction process is mathematically perfect, relying on a lossless video codec to ensure zero data degradation.

## ✨ Key Features
* **Military-Grade Encryption**: Secures the secret video using AES-GCM with PBKDF2 key derivation and randomized salts.
* **Lossless Steganography**: Utilizes the FFV1 lossless codec (`.mkv` format) to ensure pixel-perfect embedding, guaranteeing the secret video can be recovered without corruption.
* **Dynamic Frame Processing**: Automatically resizes secret frames to match the cover video and loops the secret video if it is shorter than the cover.
* **Real-Time Quality Metrics**: Calculates Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE) dynamically during processing.
* **Graphical User Interface (GUI)**: Intuitive Tkinter-based interface for the encryption and embedding phases.

## 📂 Project Structure
* `video_hiding.py` - The GUI application for AES encryption and alpha-blended video steganography.
* `video_extraction.py` - The command-line script to subtract the cover video and extract the hidden data perfectly.
* `stego_output.mkv` - The final video containing the hidden data (generated automatically).
* `stego_encrypted.bin` - Temporary binary file for AES encrypted data.

## 🧠 How it Works
1. **Phase 1 (Encryption):** The secret video is scrambled into an unrecognizable binary file (`.bin`) using a user-defined password.
2. **Phase 2 (Embedding):** The binary is decrypted in memory. Each frame of the secret video is multiplied by an `alpha` value (default `0.04`) to make it nearly transparent, then added to the cover video frames.
3. **Phase 3 (Extraction):** The receiver takes the stego video and the original cover video. The script subtracts the cover video pixels from the stego video and divides by the `alpha` value to perfectly reconstruct the secret frames.


https://github.com/user-attachments/assets/f959da78-9e46-4f91-bfb0-07b04b880e02




![1745126575015](https://github.com/user-attachments/assets/3558e59f-c439-4255-9c8d-84995e510097)
![1745126583186](https://github.com/user-attachments/assets/78d30d79-9901-4fe7-a378-88e28f8c1c49)
