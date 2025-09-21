# import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image, ImageFilter

from diffusers import FluxFillPipeline


import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 50

TARGET_RES = 1024

logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def resize_dimensions(dimensions, target_size):
    w, h = dimensions
    if w < target_size and h < target_size:
        return dimensions
    if w > h:
        ar = h / w
        return target_size, int(target_size * ar)
    ar = w / h
    return int(target_size * ar), target_size


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=16):
    return max(m, (x // m) * m)


def compute_work_resolution(w, h, max_side=1024):
    scale = min(max_side / max(w, h), 1.0)
    new_w = round_to_multiple(int(w * scale), 16)
    new_h = round_to_multiple(int(h * scale), 16)
    return new_w, new_h


def prepare_mask(mask_img: Image.Image,
                 size: tuple[int, int],
                 radius: int = 0,
                 blur: float = 0.0,
                 threshold: int = 128,
                 hard_binarize_before: bool = True) -> Image.Image:
    """
    Подготовка маски для инпейнта.
    Белое (255) = перерисовать, чёрное (0) = сохранить.  ⟵
    поведение Diffusers/Flux Fill radius > 0 -> диляция 
    (расширяем белое), radius < 0 -> эрозия (сужаем белое).
    blur > 0 -> смягчаем край маски гауссовым блюром (перо).

    Параметры:
      size   : (W, H) конечного ворк-размера
      radius : пиксели; >0 dilate, <0 erode, 0 — без изменений
      blur   : пиксели GaussianBlur для мягкого края
      threshold: порог предварительной бинаризации
      hard_binarize_before: сначала почистить маску до 0/255, затем применять
    """
    m = mask_img.convert("L").resize(size, Image.Resampling.LANCZOS)

    # 1) Чистим артефакты кисти/anti-alias (опц.)
    if hard_binarize_before:
        m = m.point(lambda p: 255 if p >= threshold else 0)

    # 2) Морфология через PIL-фильтры (без OpenCV-зависимостей)
    r = int(radius)
    if r != 0:
        k = max(1, 2 * abs(r) + 1)  # ядро нечётного размера
        if r > 0:
            # ДИЛЯЦИЯ белого: расширяем область для перерисовки
            m = m.filter(ImageFilter.MaxFilter(k))
        else:
            # ЭРОЗИЯ белого: сужаем область для перерисовки
            m = m.filter(ImageFilter.MinFilter(k))
    # 3) Мягкий край (перо): после морфологии НЕ перебинаризуем — оставляем
    if blur and float(blur) > 0.0:
        m = m.filter(ImageFilter.GaussianBlur(float(blur)))
    return m


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
repo_id = "black-forest-labs/FLUX.1-Fill-dev"
PIPELINE = FluxFillPipeline.from_pretrained(
    repo_id,
    torch_dtype=DTYPE
).to(DEVICE)


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        mask_url = payload.get("mask_url")
        if not image_url or not mask_url:
            return {"error": "'image_url' is required"}
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        guidance_scale = float(payload.get(
            "guidance_scale", 3.5))

        mask_radius = int(payload.get("mask_radius", 0))
        mask_blur = float(payload.get("mask_blur", 0.0))

        steps = min(int(payload.get(
            "steps", MAX_STEPS)),
                    MAX_STEPS)

        seed = int(payload.get(
            "seed",
            random.randint(0, MAX_SEED)))
        # generator = torch.Generator(
        #     device="cpu").manual_seed(seed)
        generator = (torch.Generator(
            device=DEVICE) if DEVICE == "cuda" else torch.Generator()
                     ).manual_seed(seed)

        image_pil = url_to_pil(image_url)

        orig_w, orig_h = image_pil.size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)

        # resize *both* init image and  control image to same, /8-aligned size
        image_pil = image_pil.resize((work_w, work_h),
                                     Image.Resampling.LANCZOS)
        mask_pil = url_to_pil(mask_url)
        mask_image = prepare_mask(
            mask_pil, (work_w, work_h),
            radius=mask_radius,
            blur=mask_blur,
            threshold=int(payload.get("mask_threshold", 128)),
            hard_binarize_before=bool(payload.get("mask_hard_binarize_before",
                                                  True))
        )
        # mask_image = mask_image.resize((work_w, work_h),
        #                                Image.Resampling.LANCZOS)
        # ------------------ генерация ---------------- #
        with torch.inference_mode():
            images = PIPELINE(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                width=work_w,
                height=work_h,
                max_sequence_length=int(payload.get(
                    "max_sequence_length", 512)),
            ).images

        return {
            "images_base64": [pil_to_b64(i) for i in images],
            "time": round(time.time() - job["created"],
                          2) if "created" in job else None,
            "steps": steps, "seed": seed
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."} # noqa
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
