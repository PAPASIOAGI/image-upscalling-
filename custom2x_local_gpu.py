import os
import sys
import time
import urllib.request
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")

if os.path.exists(VENV_PYTHON) and os.environ.get("CUSTOM2X_LAUNCHED") != "1":
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(VENV_PYTHON)):
        import subprocess
        print(f"[Custom2x] Switching to venv: {VENV_PYTHON}", flush=True)
        env = os.environ.copy()
        env["CUSTOM2X_LAUNCHED"] = "1"
        sys.exit(subprocess.call([VENV_PYTHON, "-u", os.path.abspath(__file__)] + sys.argv[1:], env=env))

# basicsr patch for torchvision >= 0.17
try:
    import torchvision.transforms.functional as _tv_f
    sys.modules["torchvision.transforms.functional_tensor"] = _tv_f
    del _tv_f
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"[Custom2x] Running on {DEVICE} ({GPU_NAME})")

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "Anime (RealESRGAN_x4plus_anime_6B)": {
        "arch": "rrdbnet",
        "num_block": 6,
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "supports_denoise": False,
    },
    "General (realesr-general-x4v3)": {
        "arch": "srvgg",
        "num_conv": 32,
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "filename": "realesr-general-x4v3.pth",
        "supports_denoise": True,
        "wdn_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "wdn_filename": "realesr-general-wdn-x4v3.pth",
    },
}

_current_key = None
_current_upsampler = None


def download_weight(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000:
        return

    tmp_path = dest_path + ".tmp"
    filename = os.path.basename(dest_path)
    print(f"[Custom2x] Downloading {filename}...", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_path)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(tmp_path, dest_path)
        print(f"[Custom2x] Download finished: {filename}", flush=True)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed downloading {filename}: {e}")


def get_upsampler(model_key: str, denoise_strength: float):
    global _current_key, _current_upsampler

    cfg = MODEL_CONFIGS[model_key]
    effective_denoise = round(denoise_strength, 2) if cfg.get("supports_denoise", False) else 0.0
    cache_key = (model_key, effective_denoise)

    if _current_key == cache_key and _current_upsampler is not None:
        return _current_upsampler

    from realesrgan import RealESRGANer

    if _current_upsampler is not None:
        del _current_upsampler
        _current_upsampler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    weight_path = os.path.join(WEIGHTS_DIR, cfg["filename"])
    download_weight(cfg["url"], weight_path)

    model_path = weight_path
    dni_weight = None

    if cfg["arch"] == "rrdbnet":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=cfg["num_block"],
            num_grow_ch=32,
            scale=cfg["netscale"],
        )
    else:
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=cfg["num_conv"],
            upscale=cfg["netscale"],
            act_type="prelu",
        )

        if cfg["supports_denoise"] and effective_denoise < 1.0:
            wdn_path = os.path.join(WEIGHTS_DIR, cfg["wdn_filename"])
            download_weight(cfg["wdn_url"], wdn_path)
            model_path = [weight_path, wdn_path]
            dni_weight = [effective_denoise, 1.0 - effective_denoise]

    upsampler = RealESRGANer(
        scale=cfg["netscale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=(DEVICE == "cuda"),
        device=DEVICE,
    )

    _current_key = cache_key
    _current_upsampler = upsampler
    return upsampler


def export_image(image: Image.Image, format_pilihan: str) -> str:
    if image is None:
        return None

    fmt = str(format_pilihan).upper().strip()
    ext = ".jpg" if fmt in ("JPG", "JPEG") else f".{fmt.lower()}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUTS_DIR, f"custom2x_{timestamp}{ext}")

    if fmt in ("JPG", "JPEG"):
        if image.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", image.size, (255, 255, 255))
            rgba = image.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            bg.save(filepath, format="JPEG", quality=95, optimize=True)
        else:
            image.convert("RGB").save(filepath, format="JPEG", quality=95, optimize=True)
    elif fmt == "WEBP":
        image.save(filepath, format="WEBP", quality=95)
    else:
        image.save(filepath, format="PNG", optimize=True)

    return filepath


def on_format_change(last_image, fmt):
    if last_image is None:
        return None
    return export_image(last_image, fmt)


def proses_gambar(gambar_input, model_pilihan, scale_pilihan, denoise_strength, format_unduhan="PNG"):
    if gambar_input is None:
        raise gr.Error("Silakan upload gambar terlebih dahulu.")

    t_start = time.time()
    has_alpha = (gambar_input.mode == "RGBA")
    image = gambar_input if has_alpha else gambar_input.convert("RGB")
    outscale = int(scale_pilihan.replace("x", ""))

    try:
        upsampler = get_upsampler(model_pilihan, denoise_strength)
    except ImportError as e:
        raise gr.Error(f"Dependensi Real-ESRGAN belum siap: {e}")
    except Exception as e:
        raise gr.Error(f"Gagal memuat model: {e}")

    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGB2BGR)

    try:
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error("VRAM penuh (OOM). Gunakan resolusi lebih kecil atau tile=256.")
        raise gr.Error(f"Gagal upscale: {e}")

    if has_alpha and output_bgr.shape[2] == 4:
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGRA2RGBA)
    else:
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    hasil = Image.fromarray(output_rgb)
    file_unduhan = export_image(hasil, format_unduhan)
    elapsed = time.time() - t_start

    info_txt = (
        f"Input      : {image.width} x {image.height} px\n"
        f"Output     : {hasil.width} x {hasil.height} px\n"
        f"Skala      : {scale_pilihan}\n"
        f"Format     : {str(format_unduhan).upper()}\n"
        f"Waktu      : {elapsed:.2f} detik\n"
        f"Model      : {model_pilihan}\n"
        f"Device     : {DEVICE.upper()} ({GPU_NAME})"
    )

    return (image, hasil), info_txt, file_unduhan, hasil


def on_model_changed(model_name):
    supports_denoise = MODEL_CONFIGS.get(model_name, {}).get("supports_denoise", False)
    return gr.update(interactive=supports_denoise, value=0.5 if supports_denoise else 0.0)


with gr.Blocks(title="Custom2x - AI Image Upscaler") as demo:
    gr.Markdown(
        f"""
        # Custom2x - AI Image Upscaler
        Local GPU image upscaling with Real-ESRGAN.
        Device: **{DEVICE.upper()}** - `{GPU_NAME}`
        """
    )

    hasil_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input & Pengaturan")

            input_image = gr.Image(label="Upload Gambar", type="pil")

            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="Anime (RealESRGAN_x4plus_anime_6B)",
                label="Pilih Model",
                info="Anime: untuk 2D & ilustrasi | General: untuk foto realistik",
            )

            scale_dropdown = gr.Radio(
                choices=["2x", "4x"],
                value="4x",
                label="Skala Upscaling",
            )

            denoise_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.05,
                label="Kekuatan Denoise",
                info="Khusus model General (0.0 = pertahankan grain asli)",
                interactive=False,
            )

            format_dropdown = gr.Radio(
                choices=["PNG", "JPG", "WEBP"],
                value="PNG",
                label="Format Unduhan",
                info="PNG (Lossless & Alpha) | JPG (Kecil) | WEBP (Modern & Ringan)",
            )

            tombol_proses = gr.Button("Enhance Resolution", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Hasil Perbandingan")

            output_slider = gr.ImageSlider(
                label="Sebelum vs Sesudah",
                type="pil",
                buttons=["fullscreen"],
            )
            download_file = gr.File(label="Unduh Gambar Hasil", interactive=False)
            info_box = gr.Textbox(label="Informasi Proses", lines=7, interactive=False)

    model_dropdown.change(
        fn=on_model_changed,
        inputs=[model_dropdown],
        outputs=[denoise_slider],
    )

    format_dropdown.change(
        fn=on_format_change,
        inputs=[hasil_state, format_dropdown],
        outputs=[download_file],
    )

    tombol_proses.click(
        fn=proses_gambar,
        inputs=[input_image, model_dropdown, scale_dropdown, denoise_slider, format_dropdown],
        outputs=[output_slider, info_box, download_file, hasil_state],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
