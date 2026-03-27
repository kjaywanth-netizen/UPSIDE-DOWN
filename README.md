# UPSIDE-DOWN
# Offroad Semantic Segmentation

This repository contains our team's submission for the Duality AI Hawk-A-Thon (Offroad Segmentation Track). It implements a DINOv2-based architecture with advanced Feature Pyramid decoding designed for real-time 83+ FPS inference on Unmanned Ground Vehicles (UGVs).

## 🚀 Environment Setup
To reproduce our results on a native Windows machine with an NVIDIA GPU (e.g., RTX 4050), follow these steps to set up the correct PyTorch CUDA environment:

1. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. **Install PyTorch with CUDA 12.1**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Install Core Dependencies**
   ```bash
   pip install opencv-python matplotlib numpy tqdm pillow albumentations
   ```
*(Note for Windows RTX 4050 Users: Standard `torch.compile()` with Triton is unsupported natively on Windows. Our evaluation scripts automatically bypass this limit using eager-mode fallback while preserving raw FP16 PyTorch Autocast speed).*

## 🏃‍♂️ Running the Pipeline

### 1. Training (`train_segmentation_v5.py`)
To train the model from scratch on the Falcon Synthetic Dataset:
```bash
python Offroad_Segmentation_Scripts\train_segmentation_v5.py
```
This script features an unfrozen DINOv2 backbone, an UperNet decoder head, Custom Copy-Paste Augmentation for underrepresented edge-cases, and Lovász-Softmax Loss.

### 2. Validation & Edge-Case Extraction (`visualize.py`)
To perform inference on the validation set and mathematically extract the top 3 best and worst IoU predictions:
```bash
python Offroad_Segmentation_Scripts\visualize.py
```
*Outputs high-res side-by-side PNGs to `runs/evaluation/`.*

### 3. Real-Time FPS Benchmarking (`test_inference_speed.py`)
To verify the 11.97ms / 83.55 FPS real-time speed metric on your GPU architecture:
```bash
python test_inference_speed.py
```
