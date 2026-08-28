# *************key控制DDIM实现确定图像生成（跨设备一致性增强版）*************
from typing import Optional, Union, Tuple, List, Callable, Dict
import os
import random
import hashlib
import argparse
import json
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# =========================
# 可复现 / 确定性相关设置
# =========================
def configure_determinism():
    """
    尽量在创建 CUDA 上下文前调用。
    注意：这会降低性能，但有助于提升跨设备（一致配置）复现性。
    """
    # cuBLAS 工作区配置（GPU确定性常用）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # OpenCV 线程固定（避免一些平台差异）
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # PyTorch 确定性算法（严格模式）
    torch.use_deterministic_algorithms(True)

    # 固定线程数，减少并行归约导致的非确定性
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # 某些环境重复设置会报错，可忽略
        pass

    # cuDNN / TF32 设置
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    # 尽量关闭可能引入路径差异的高性能 SDPA 内核（如果版本支持）
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        fn = getattr(torch.backends.cuda, "enable_flash_sdp", None)
        if callable(fn):
            fn(False)
        fn = getattr(torch.backends.cuda, "enable_mem_efficient_sdp", None)
        if callable(fn):
            fn(False)
        fn = getattr(torch.backends.cuda, "enable_math_sdp", None)
        if callable(fn):
            fn(True)


def seed_everything(seed: int):
    """固定所有常见随机源。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pixel_sha256(img_rgb_uint8: np.ndarray) -> str:
    """对 HWC uint8 RGB 像素做 SHA256，用于逐像素一致性校验。"""
    arr = np.ascontiguousarray(img_rgb_uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """对张量内容做 SHA256（先转 CPU contiguous bytes）。"""
    x = t.detach().to("cpu").contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()

def quantize_latents(latents: torch.Tensor, q: float) -> torch.Tensor:
    """
    强制“收敛微差”：对 latent 做定点化/量化。
    - q=1e4 表示保留 1e-4 精度：round(latents*q)/q
    - q=1e3 表示保留 1e-3 精度
    说明：该操作会改变生成分布，但通常能显著降低跨环境的像素差异扩散。
    """
    if q is None or q <= 0:
        return latents
    # 保持 float32，避免不同精度路径
    if latents.dtype != torch.float32:
        latents = latents.to(dtype=torch.float32)
    return torch.round(latents * q) / q



# 先配置确定性，再初始化模型
configure_determinism()

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)

MY_TOKEN = ''
LOW_RESOURCE = False
GUIDANCE_SCALE = 1.0
MAX_NUM_WORDS = 77

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 为增强一致性，建议固定 FP32（会增加显存占用）
ldm_stable = StableDiffusionPipeline.from_pretrained(
    "D:/code/IS/CRoSS-main/runwayml/stable-diffusion-v1-5",
    use_auth_token=MY_TOKEN,
    scheduler=scheduler,
    torch_dtype=torch.float32
).to(device)

# 显式 eval（通常 pipeline 已是 eval，这里再次确认）
try:
    ldm_stable.unet.eval()
    ldm_stable.vae.eval()
    ldm_stable.text_encoder.eval()
except Exception:
    pass

# 关闭 xformers memory efficient attention（若存在）
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")

tokenizer = ldm_stable.tokenizer


def load_image(image_path, left=0, right=0, top=0, bottom=0, resize=False):
    image = Image.open(image_path).convert("RGB")
    h, w = image.size  # 注意：PIL size 返回 (W,H)，这里沿用原代码变量命名
    if resize:
        image = np.array(image.resize((512, 512)))
    else:
        width_padding = -h % 8
        height_padding = -w % 8

        padded_image = ImageOps.expand(
            image,
            (0, 0, width_padding, height_padding),
            fill=None,
        )
        image = np.array(padded_image)
    return image, h, w


class ODESolve:

    def __init__(self, model, NUM_DDIM_STEPS=50, latent_quant: float = 0.0):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.latent_quant = float(latent_quant) if latent_quant is not None else 0.0

        # 某些 diffusers 版本支持 device 参数，做兼容
        try:
            self.model.scheduler.set_timesteps(self.num_ddim_steps, device=self.model.device)
        except TypeError:
            self.model.scheduler.set_timesteps(self.num_ddim_steps)

        self.prompt = None
        self.context = None

    def prev_step(self,
                  model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self,
                  model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999
        ), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        if context is None:
            context = self.context
        guidance_scale = GUIDANCE_SCALE
        uncond_embeddings, cond_embeddings = context.chunk(2)

        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = (1 / 0.18215) * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]

        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]

        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        return text_embeddings

    @torch.no_grad()
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            # 强制收敛微差：每一步对 latent 定点化/量化
            latent = quantize_latents(latent, self.latent_quant)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def save_inter(self, latent, img_name):
        image = self.latent2image(latent)
        cv2.imwrite(img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def invert(self, prompt, start_latent, is_forward):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]

    def invert_i2n2i(self, prompt1, prompt2, image_start_latent, flip=False):

        self.init_prompt(prompt1)
        latent_i2n = self.ddim_loop(image_start_latent, is_forward=True)
        xT = latent_i2n[-1]

        if flip:
            xT = torch.flip(xT, dims=[2])

        self.init_prompt(prompt2)
        latent_n2i = self.ddim_loop(xT, is_forward=False)

        return (
            self.latent2image(image_start_latent),
            image_start_latent,
            self.latent2image(latent_n2i[-1]),
            latent_n2i[-1]
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./result2/ddim2_batch1000_env2_g1.0_0_3_3_10000_lx', help='output directory')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling steps of DDIM')
    parser.add_argument('--latent_quant', type=float, default=0,
                        help='Latent quantization scale q. Each step: latents=round(latents*q)/q. '
                             'Use 0 to disable. Typical: 1e4 (1e-4 precision) or 1e3 (1e-3).')
    parser.add_argument(
        '--prompts_json', type=str, default='./prompts_10000_key_32bit_10000.json',
        help='JSON mapping output filenames to {prompt, key}. Example: {"0001.png": {"prompt": "...", "key": 123...}, ...}'
    )
    parser.add_argument('--height', type=int, default=512, help='generation height (must be multiple of 8)')
    parser.add_argument('--width', type=int, default=512, help='generation width (must be multiple of 8)')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # -------------------------
    # Load prompts + fixed keys (Attachment 2)
    # -------------------------
    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)

    # Stable ordering by filename (e.g., 0001.png ... 0010.png)
    items = sorted(prompts_dict.items(), key=lambda x: x[0])

    ode = ODESolve(ldm_stable, args.num_steps, latent_quant=args.latent_quant)

    # Fixed generation resolution
    height, width = args.height, args.width
    if (height % 8) != 0 or (width % 8) != 0:
        raise ValueError("height/width must be multiples of 8 (e.g., 512).")

    latent_shape = (
        1,
        ldm_stable.unet.in_channels,
        height // ldm_stable.vae_scale_factor,
        width // ldm_stable.vae_scale_factor,
    )

    # Batch log
    log_path = os.path.join(args.save_path, "batch_repro_info.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("filename,key_uint64,seed_int63,num_steps,latent_quant,latent0_sha256,pixel_sha256,prompt\n")

    for filename, entry in items:
        # entry format:
        #   - {"prompt": "...", "key": <uint64 decimal int>}   (recommended)
        #   - "prompt string"                                 (fallback; NOT recommended for your use case)
        if isinstance(entry, str):
            prompt = entry
            raise ValueError(
                f'JSON entry for "{filename}" is a plain string (prompt only). '
                f'Please provide a fixed per-image key: {{"{filename}": {{"prompt": "...", "key": 123...}}}}'
            )
        elif isinstance(entry, dict):
            if "prompt" not in entry or "key" not in entry:
                raise ValueError(
                    f'JSON entry for "{filename}" must contain both "prompt" and "key". Got: {entry}'
                )
            prompt = entry["prompt"]
            key_uint64 = int(entry["key"])
        else:
            raise ValueError(f'Unsupported JSON entry type for "{filename}": {type(entry)}')

        # ------------------------------------------------------------
        # Deterministic seed per image from JSON key:
        #   - torch seeds are safest in [0, 2^63-1] across platforms
        # ------------------------------------------------------------
        seed = key_uint64 & ((1 << 63) - 1)

        # Fix all RNGs (Python / NumPy / Torch)
        seed_everything(seed)

        # CPU generator for initial noise (reduces cross-device RNG differences)
        gen_cpu = torch.Generator(device="cpu")
        gen_cpu.manual_seed(seed)

        latent_noise_cpu = torch.randn(
            latent_shape,
            generator=gen_cpu,
            device="cpu",
            dtype=torch.float32
        )

        latent0_hash = tensor_sha256(latent_noise_cpu)

        # Move to target device (keep FP32)
        latent_noise = latent_noise_cpu.to(device=device, dtype=torch.float32).contiguous()

        # DDIM sampling (noise -> image)
        image_latent = ode.invert(prompt, latent_noise, is_forward=False)
        image_rgb = ode.latent2image(image_latent)

        pix_hash = pixel_sha256(image_rgb)

        # Save image
        out_path = os.path.join(args.save_path, filename)
        cv2.imwrite(out_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        # Append log (CSV)
        prompt_csv = str(prompt).replace('"', '""')
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f'{filename},{key_uint64},{seed},{args.num_steps},{args.latent_quant},{latent0_hash},{pix_hash},"{prompt_csv}"\n')

        # Optional: per-image hash files (handy for debugging)
        with open(os.path.join(args.save_path, f"{os.path.splitext(filename)[0]}.latent0.sha256.txt"), "w", encoding="utf-8") as f:
            f.write(latent0_hash + "\n")
        with open(os.path.join(args.save_path, f"{os.path.splitext(filename)[0]}.pixel.sha256.txt"), "w", encoding="utf-8") as f:
            f.write(pix_hash + "\n")

        print(f"[OK] {filename} | key={key_uint64} | seed={seed} | latent0={latent0_hash[:10]}... | pixel={pix_hash[:10]}...")

    print(f"Done. Images saved to: {args.save_path}")
    print(f"Batch log saved to: {log_path}")
