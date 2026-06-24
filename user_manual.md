# 📖 Comprehensive User Manual: Video Steganography via Transform Domain

Welcome to the Video Steganography tool. This software allows you to securely hide a secret video inside a normal, "innocent-looking" cover video using Alpha Blending and military-grade AES-256 encryption.

This manual is divided into three sections:
1. **Prerequisites & Setup**
2. **Phase 1: Hiding the Secret Video (GUI)**
3. **Phase 2: Extracting the Secret Video (CLI)**
4. **Understanding Your Results & Metrics**
5. **Troubleshooting Guide**

---

## 🛠️ 1. Prerequisites & Setup

Before you begin, ensure your system is prepared.

### System Requirements
* Python 3.7 or higher installed on your computer.
* Required Python libraries installed: `opencv-python`, `numpy`, `pycryptodome`, and `scikit-image`.

### Preparing Your Files
You must have two video files ready in the same folder as your scripts:
1. **The Cover Video** (e.g., `cover.mp4`): The standard video that will act as the "carrier." Anyone watching this will just see a normal video.
2. **The Secret Video** (e.g., `secret.mp4`): The private video you want to hide.

*Tip: For the best results, ensure both videos are functional and play smoothly in standard media players (like VLC or Windows Media Player) before starting.*

---

## 🔒 2. Phase 1: Hiding the Secret Video

The hiding process is handled entirely through an easy-to-use Graphical User Interface (GUI).

### Step 1: Launch the Application
Open your terminal or command prompt, navigate to your project folder, and run:
```bash
python video_hiding.py
```

A window titled **"Video In Video Hiding"** will appear.

### Step 2: Select Your Media

1. Click the **1. Select Cover Video** button. A file browser will open. Locate and select your innocent cover video.
2. Click the **2. Select Secret Video** button. Locate and select the video you wish to hide.

### Step 3: Encrypt the Secret Video

Before hiding, the video is scrambled to ensure that even if someone detects it, they cannot view it without the password.

1. In the **Set Encryption Key** text box, type a strong, memorable password.
2. Click the **3. Process & Encrypt Secret Video** button.
3. Wait a moment. A popup box saying "Encryption Complete" will appear, and a temporary `stego_encrypted.bin` file will be created in your folder.

### Step 4: Embed the Video

1. In the lower text box labeled **Enter Key to Decrypt & Embed**, type the **exact same password** you used in Step 3.
2. Click the **4. Embed Stego Video** button.
3. The process will begin. Watch the text console at the bottom of the window; it will update you frame-by-frame.
* *Note: This process is computationally heavy and uses lossless FFV1 encoding to ensure the hidden data survives. It may take several minutes depending on video length and computer speed.*
4. Once complete, a detailed statistical report will print on the screen.
5. Your final, safe-to-share file is saved in your project directory as **`stego_output.mkv`**.

---

## 🔓 3. Phase 2: Extracting the Secret Video

When you or the intended recipient wants to view the hidden video, you will use the extraction script. **You must have the original, unmodified cover video to perform extraction.**

### Step 1: Verify Your Files

Ensure these three files are in the exact same folder:

1. `video_extraction.py`
2. `stego_output.mkv` (The file generated in Phase 1)
3. The exact, original cover video (e.g., `cover.mp4`) used in Phase 1.

### Step 2: Update the Extraction Script (If necessary)

Open `video_extraction.py` in any text editor (like Notepad or VS Code) and scroll to the very bottom:

```python
if __name__ == "__main__":
    stego_file = "stego_output.mkv"
    cover_file = "cover.mp4" # <-- MAKE SURE THIS MATCHES YOUR COVER VIDEO EXACTLY
    output_file = "extracted_secret.avi"
```

Ensure the `cover_file` variable exactly matches the name and extension of your cover video. Save and close the file.

### Step 3: Run the Extraction

Open your terminal and run:

```bash
python video_extraction.py
```

The terminal will display the frame-by-frame extraction progress. Once it reaches the end, you will find a new file named **`extracted_secret.avi`** in your folder. Double-click it to watch your recovered secret video!

---

## 📊 4. Understanding Your Results & Quality Metrics

At the end of Phase 1, the software provides a detailed quality report. Here is how to read it:

* **MSE (Mean Squared Error):** Measures the average squared difference between the original cover video pixels and the final stego video pixels.
* *Ideal Score:* The lower, the better. A score near `0` means the visual changes are nearly imperceptible.
* **PSNR (Peak Signal-to-Noise Ratio):** Measures the ratio between the maximum possible power of a pixel and the power of corrupting noise (the hidden video).
* *Ideal Score:* The higher, the better. Anything above **30 dB** is considered good quality; above **40 dB** is excellent.
* **SSIM (Structural Similarity Index):** Evaluates the visual impact of changes in luminance, contrast, and structure.
* *Ideal Score:* Measured on a scale of -1 to 1. A score of **0.95 to 1.00** means the human eye will likely not notice the hidden data.

---

## 🩺 5. Troubleshooting Guide

**Error:** `ModuleNotFoundError: No module named 'Crypto'`

* **Cause:** The cryptography library is missing.
* **Fix:** Open your terminal and run: `pip install pycryptodome`. Do *not* install `pycrypto`.

**Error:** `moov atom not found` (OpenCV Error)

* **Cause:** The video file is corrupted. This usually happens if you enter the wrong decryption password during Phase 1, Step 4.
* **Fix:** Delete the `stego_encrypted.bin` file, restart the application, and ensure your passwords match perfectly.

**Error:** The extracted video is just a grey screen or static noise.

* **Cause:** The cover video used for extraction does not perfectly match the one used for hiding, OR the `stego_output.mkv` was compressed or altered after creation.
* **Fix:** Ensure you are using the exact original cover video. Do not run the `stego_output.mkv` through video compressors, WhatsApp, or standard social media platforms, as they compress videos and destroy the pixel-perfect hidden data. Transfer the `.mkv` file via USB, Google Drive, or email attachments.
