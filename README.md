# Custom2x - Local GPU Image Upscaler

A lightweight web UI for upscaling images locally using Real-ESRGAN and PyTorch with CUDA acceleration.

## Features

- **Local Inference**: Runs on your local GPU (NVIDIA CUDA) with half-precision (FP16) support.
- **Models**:
  - `Anime (RealESRGAN_x4plus_anime_6B)`: For illustrations, digital art, and anime.
  - `General (realesr-general-x4v3)`: For photos with adjustable denoise strength.
- **Scaling**: 2x and 4x upscaling.
- **Tiled Processing**: Uses 256px tiles to prevent Out of Memory (OOM) on 6GB VRAM GPUs.
- **Multi-Format Export**: Download directly as `PNG`, `JPG`, or `WEBP`.
- **Instant Format Switch**: Re-export already upscaled images without re-running GPU inference.
- **Interactive Comparison**: Before/after image comparison slider.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on GTX 1060 6GB)
- Windows / Linux

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PAPASIOAGI/image-upscalling-.git
   cd image-upscalling-
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Make sure PyTorch is installed with CUDA support from [pytorch.org](https://pytorch.org))*

## Usage

Start the app:
```bash
python custom2x_local_gpu.py
```
Or run `jalankan_custom2x_gpu.bat` on Windows.

Model weights will be downloaded automatically to the `weights/` folder on first use.

## License

MIT
