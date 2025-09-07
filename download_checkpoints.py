import os
import torch

from diffusers import FluxFillPipeline

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ------------------------- пайплайн -------------------------
def get_pipeline():
    repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    FluxFillPipeline.from_pretrained(repo_id,
                                     torch_dtype=torch.bfloat16
                                     ).to(DEVICE)


if __name__ == "__main__":
    get_pipeline()
