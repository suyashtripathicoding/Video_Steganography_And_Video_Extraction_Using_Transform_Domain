# System Requirements & Dependencies

This project requires **Python 3.7+**. The core operations rely on computer vision and cryptography libraries. 

## 📦 Required Python Libraries

* **`opencv-python`**: Used for reading, processing, resizing, and writing video frames.
* **`numpy`**: Used for heavy-duty multi-dimensional matrix math (subtracting and blending frames).
* **`pycryptodome`**: Handles the AES-256 GCM encryption and PBKDF2 key generation.
* **`scikit-image`**: Used exclusively to calculate the Structural Similarity (SSIM) metric.

*(Note: `tkinter`, `os`, and `time` are built into standard Python and do not require installation.)*

## 🛠️ Installation Instructions

To install all dependencies and avoid common namespace conflicts with legacy cryptography libraries, run the following command in your terminal or VS Code environment:

```powershell
python -m pip uninstall crypto pycrypto pycryptodome -y
python -m pip install pycryptodome opencv-python numpy scikit-image
