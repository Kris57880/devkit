from utils.degradations import VVC, MS_ILLM, ELIC
from utils.custom_sd_pipeline import StableDiffusionPGDMPipeline, StableDiffusionNCPipeline, StableDiffusionMPGDPipeline, StableDiffusionVAEOnlyPipeline, StableDiffusionNormalPipeline
import torchvision.transforms as transforms

from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import torch
import os
import PIL
from functools import partial
import argparse
from tqdm import tqdm
from tiler import Tiler, Merger
import random
import pandas as pd

def set_all_seeds(seed=1024):
    """Fix all possible sources of randomness"""
    # Python random module
    random.seed(seed)
    
    # NumPy randomness
    np.random.seed(seed)
    
    # PyTorch randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU support
    
    # CuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch other settings
    torch.use_deterministic_algorithms(True)
    
    # Set environment variables to ensure complete determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_captions_from_csv(csv_path, image_name):
    """
    從 CSV 文件中載入特定圖像的所有 patch captions
    
    Args:
        csv_path (str): CSV 文件路徑
        image_name (str): 圖像名稱 (不含副檔名)
    
    Returns:
        dict: {patch_index: caption} 的字典
    """
    try:
        df = pd.read_csv(csv_path)
        # 過濾出對應圖像的 captions
        image_captions = df[df['image_name'] == image_name]
        
        # 建立 patch_index 到 caption 的映射
        captions_dict = {}
        for _, row in image_captions.iterrows():
            captions_dict[int(row['patch_index'])] = row['caption']
        
        return captions_dict
    except Exception as e:
        print(f"載入 CSV 文件時發生錯誤: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description="BK-SDM denim.py runner")
    parser.add_argument('-i','--input_folder', type=str, default='', help='Path to input directory')
    parser.add_argument('-o','--output_folder', type=str, default='../results', help='Path to save output directory')
    parser.add_argument('--model_path', type=str, default='nota-ai/bk-sdm-v2-base', help='Path to the pre-trained model')
    parser.add_argument('--guidance_scale', type=float, default=0, help='Guidance scale for classifier-free guidance')
    # setting for denim 2 stages
    parser.add_argument('-r','--rate', type=float, default=0.075, help='choose from 0.075, 0.15, 0.3 (bpp)') 
    # stage 1: noise calibration
    parser.add_argument('--noise_regularization', action='store_true', help='Whether to apply noise regularization')
    parser.add_argument('--N_start_reg', type=int, default=0, help='Start step for noise regularization')
    # stage 2: PGDM/MPGD/Unconditional
    parser.add_argument('--mode', type=str, default='pgdm', help='Mode: pgdm, mpgd, normal')
    parser.add_argument('--grad_term_weight', type=float, default=0.01, help='Weight for the gradient term in PGDM/MPGD')
    return parser.parse_args()

class PatchSplitter:
    def __init__(self, tile_shape=(512, 512), overlap=0.25):
        """
        Initialize the splitter using the tiler library.

        Args:
            tile_shape (tuple): (height, width) of each patch. Default is (512, 512).
            overlap (float): Overlap ratio between adjacent patches. Default is 0.25 (25%).
                             This is the key to solving block artifacts.
        """
        if not (0 <= overlap < 1):
            raise ValueError("Overlap must be in the range [0, 1)")
            
        self.tile_h, self.tile_w = tile_shape
        self.overlap = overlap
        
        # These variables will be set in split_image for use in merge_patches
        self.tilers = None
        self.mergers = None
        self.num_tiles_per_image = None
        self.original_torch_info = {}

    def split_image(self, src: torch.Tensor) -> torch.Tensor:
        """
        Split input image into overlapping patches using Tiler.

        Args:
            src (torch.Tensor): Input image tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: A tensor containing all patches,
                          with shape (total patches count, C, tile_h, tile_w).
        """
        # Check if input is PyTorch Tensor
        if not isinstance(src, torch.Tensor):
            raise TypeError("Input 'src' must be a PyTorch Tensor.")
            
        # Store original tensor information for restoration during merging
        self.batch_size, C, H, W = src.shape
        self.original_torch_info = {
            'device': src.device,
            'dtype': src.dtype,
            'batch_size': self.batch_size
        }
        
        # Tiler processes single images (C, H, W), so we need to iterate through batch
        self.tilers = []
        self.mergers = []
        self.num_tiles_per_image = []
        all_patches_list = []

        for i in range(self.batch_size):
            # Extract single image from batch and convert to numpy format (C, H, W)
            img_np = src[i].detach().cpu().numpy()
            
            # Create Tiler and corresponding Merger
            tiler = Tiler(
                data_shape=img_np.shape,
                tile_shape=(C, self.tile_h, self.tile_w),
                overlap=self.overlap,
                channel_dimension=0,   # C is the first dimension
                mode='reflect' # handling edge tiles 
            )
            merger = Merger(tiler, window= "hann")
            
            self.tilers.append(tiler)
            self.mergers.append(merger)
            self.num_tiles_per_image.append(len(tiler))
            
            # Iterate to generate patches and collect them
            for _, tile_np in tiler.iterate(img_np):
                all_patches_list.append(torch.from_numpy(tile_np))

        # Stack all patches into a large tensor
        if not all_patches_list:
            # If image is too small to generate any patches, return empty Tensor
            return torch.empty(0, C, self.tile_h, self.tile_w, 
                               device=self.original_torch_info['device'],
                               dtype=self.original_torch_info['dtype'])

        patches_tensor = torch.stack(all_patches_list, dim=0).to(
            device=self.original_torch_info['device'],
            dtype=self.original_torch_info['dtype']
        )
        
        return patches_tensor

    def merge_patches(self, patches) :
        """
        (Modified) Use Merger to seamlessly merge patches list back to original image dimensions.

        Args:
            patches (List[torch.Tensor]): List of processed patch tensors.
                                           Each element in the list is a patch tensor.

        Returns:
            torch.Tensor: Merged image tensor with shape (B, C, H, W).
        """
        if self.tilers is None or self.mergers is None:
            raise RuntimeError("You must call split_image before merge_patches.")
        
        # Check if patches list length is correct
        expected_total_patches = sum(self.num_tiles_per_image)
        if len(patches) != expected_total_patches:
            raise ValueError(f"Expected {expected_total_patches} patches, but got {len(patches)}.")

        merged_images = []
        current_patch_idx = 0

        for i in range(self.original_torch_info['batch_size']):
            merger = self.mergers[i]
            num_tiles = self.num_tiles_per_image[i]
            
            # Extract corresponding patches from list and add to merger
            for tile_id in range(num_tiles):
                # Get patch tensor directly from list and remove batch dimension
                patch_tensor = patches[current_patch_idx]
                # Remove batch dimension (1, C, H, W) -> (C, H, W)
                if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
                    patch_tensor = patch_tensor.squeeze(0)
                merger.add(tile_id, patch_tensor.cpu().numpy())
                current_patch_idx += 1

            merged_np = merger.merge(unpad=True)
            merged_images.append(torch.from_numpy(merged_np))

        final_batch = torch.stack(merged_images, dim=0)
        final_batch = final_batch.to(
            device=self.original_torch_info['device'],
            dtype=self.original_torch_info['dtype']
        )
        
        return final_batch     
           

if __name__ == "__main__":
    device = "cuda"
    args = parse_args()
    if args.rate == 0.075:
        qf = 1 
    elif args.rate == 0.15:
        qf = 2
    elif args.rate == 0.3:
        qf = 4

    # initialize necessary modules
    codec = ELIC(qf)
    decoder = codec.decode
    operator = codec.H
    
    patchfier = PatchSplitter(tile_shape=(512, 512), overlap=0.25)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    pipe1 = StableDiffusionNCPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    )
    
    if args.mode == 'pgdm':
        pipe2 = StableDiffusionPGDMPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
        )
    elif args.mode == 'mpgd':
        pipe2 = StableDiffusionMPGDPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
        )
    elif args.mode == 'normal':
        pipe2 = StableDiffusionNormalPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
        )
    else :
        raise ValueError(f"Unsupported mode: {args.mode}, supported modes are 'pgdm', 'nc', 'mpgd'.")
    
    pipe1 = pipe1.to(device)
    pipe2 = pipe2.to(device)
    # decode from bitstreams
    bitstreams = []
    filenames = []
    extra_info = pd.read_csv(os.path.join(args.input_folder, 'extra_information.csv'))

    for file in os.listdir(os.path.join(args.input_folder, 'bitstream')):
        if file.endswith('.pt'):
            # print(f'Loading bitstream: {file}')
            filename = os.path.splitext(file)[0]
            filenames.append(filename)
            bitstream = torch.load(os.path.join(args.input_folder, 'bitstream', file))
            bitstreams.append(bitstream)

    images = []
    os.makedirs(os.path.join(args.output_folder,'deg'), exist_ok=True)
    for bitstream in bitstreams[:1]:
        with torch.no_grad():
            image_name = filenames[bitstreams.index(bitstream)].split('.')[0]
            h, w = extra_info[extra_info['filename'] == f"{image_name}.png"][['height', 'width']].values[0]
            image = decoder(bitstream, h, w)
            images.append(image)
            save_image(image, os.path.join(args.output_folder,"deg", f"{image_name}.png"))
            
    
    for image in images:
        image_name = filenames[images.index(image)].split('.')[0]
        step1, step2 = extra_info[extra_info['filename'] == f"{image_name}.png"][['step1', 'step2']].values[0]

        image = image*2.0 - 1.0
        image = image.to(device)
        image = image.half()  
        patches = patchfier.split_image(image)  # shape: (B*16, C, H//4, W//4)
        h, w = patches.shape[2], patches.shape[3]

        prompt = ""
        set_all_seeds(1024)
        generator = torch.Generator(device=device).manual_seed(1024)
        restored_patches = []
        for i in tqdm(range(patches.shape[0])):
            patch = patches[i:i+1]  # Keep batch dimension    
            # get prompt for this patches 
            # current_prompt = captions_dict.get(i, "")  # if no caption found, use empty string
            # print(f'patchs{i}: {current_prompt}')
            current_prompt = ""
            # Stage 1: noise calibration
            restored_patch, restored_latent = pipe1(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=patch,
                operator=operator,
                guidance_scale=args.guidance_scale , 
                start_timestep=step1, 
                # for noise calibration
                cali_N=10,
                do_noise_regularization=args.noise_regularization,
                N_start_reg=args.N_start_reg,
            )
            restored_patch = restored_patch.images[0]
            restored_latent = restored_latent.images.detach()

            restored_patch = pipe2(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=restored_latent,
                measurement_type = 'latent',
                operator=operator,
                guidance_scale=args.guidance_scale,  # unconditional
                grad_term_weight=args.grad_term_weight, 
                start_timestep=step2, 
            ).images[0]
            
            patch_tensor = torch.from_numpy(np.array(restored_patch, dtype=np.float32)).permute(2, 0, 1)[None]
            patch_tensor = (patch_tensor / 255.0)
            patch_tensor = patch_tensor.to(device)
            restored_patches.append(patch_tensor)
            
        # restored_img = shuffler.shuffle_image(restored_patches)  # (B, C, H, W)
        restored_img = patchfier.merge_patches(restored_patches)  # (B, C, H, W)
        os.makedirs(os.path.join(args.output_folder,'out'), exist_ok=True)
        save_image(restored_img ,  os.path.join(args.output_folder, "out", f"{image_name}.png"))

"""
    # 載入 captions
    captions_dict = {}
    if args.captions_csv:
        captions_dict = load_captions_from_csv(args.captions_csv, image_name)
    
    model_name = args.model_path.split('/')[-1]
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'log'), exist_ok=True)
    src = Image.open(args.input_image).convert('RGB')

    src = transform(src).unsqueeze(0)
    src = (src * 2.0 - 1.0) # Normalize to [-1, 1]
    src = src.to(device)
    src = src.half()  # Convert to half precision
    # Set up  operator
    measurement = operator(src)

    # save_image((src + 1.0) / 2.0, os.path.join(output_folder, 'ori', f"{image_name}.png"))
    # save_image((measurement + 1.0) / 2.0, os.path.join(output_folder,'deg', f"{image_name}.png"))
    model_path = args.model_path
    
    if args.mode == 'pgdm':
        pipe = StableDiffusionPGDMPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    elif args.mode =='nc': 
        pipe = StableDiffusionNCPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        os.makedirs(os.path.join(output_folder, image_name), exist_ok=True) # storing latents
    elif args.mode == 'mpgd':
        pipe = StableDiffusionMPGDPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    elif args.mode =='vae_only':
        pipe = StableDiffusionVAEOnlyPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    elif args.mode == 'normal':
        pipe = StableDiffusionNormalPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    else :
        raise ValueError(f"Unsupported mode: {args.mode}, supported modes are 'pgdm', 'nc', 'mpgd'.")
    # measurement = measurement.half()  # Convert to half precision

    # patches = shuffler.unshuffle_image(measurement)
    patches_ori = patchfier.split_image(src)  # shape: (B*16, C, H//4, W//4)
    patches = patchfier.split_image(measurement)  # shape: (B*16, C, H//4, W//4)
    # print(f"patch shapes: {patches.shape}")
    h, w = patches.shape[2], patches.shape[3]

    pipe = pipe.to(device)
    prompt = ""
    set_all_seeds(1024)
    generator = torch.Generator(device=device).manual_seed(1024)
    restored_patches = []
    # print(pipe.scheduler)
    
    for i in tqdm(range(patches.shape[0])):
        patch = patches[i:i+1]  # Keep batch dimension    
        patch_ori = patches_ori[i:i+1]
        
        # 獲取對應 patch 的 prompt
        current_prompt = captions_dict.get(i, "")  # 如果找不到對應的 caption，使用空字串
        # print(f'patchs{i}: {current_prompt}')
        if args.mode == 'nc':
            restored_patch, restored_latent = pipe(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=patch,
                operator=operator,
                guidance_scale=args.guidance_scale , 
                start_timestep=args.start_timestep, 
                log_dir=os.path.join(output_folder, 'log', f"patch_{i}"),
                # for noise calibration
                do_noise_regularization=args.noise_regularization,
                denoise_after_nc=args.denoise_after_nc,
                cali_N=10,
                original_image=patch_ori,
                N_start_reg=args.N_start_reg,
            )
            restored_patch = restored_patch.images[0]
            restored_latent = restored_latent.images[0].cpu()
        elif args.mode in ['pgdm', 'mpgd']:
            restored_patch = pipe(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=patch,
                operator=operator,
                guidance_scale=0,  # unconditional
                grad_term_weight=args.grad_term_weight, 
                start_timestep=args.start_timestep, 
                log_dir=os.path.join(output_folder, 'log', f"patch_{i}"),
            ).images[0]
        elif args.mode == 'vae_only':
            restored_patch = pipe(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=patch,
            ).images[0]
        elif args.mode == 'normal':
            restored_patch = pipe(
                prompt=current_prompt, 
                height=h,
                width=w,
                generator=generator, 
                measurement=patch,
                guidance_scale=args.guidance_scale , 
                start_timestep=args.start_timestep, 
                log_dir=os.path.join(output_folder, 'log', f"patch_{i}"),
            ).images[0]

        patch_tensor = torch.from_numpy(np.array(restored_patch, dtype=np.float32)).permute(2, 0, 1)[None]
        patch_tensor = (patch_tensor / 255.0)
        patch_tensor = patch_tensor.to(device)
        restored_patches.append(patch_tensor)
        if args.mode =='nc':
            np.save(os.path.join(output_folder, image_name, f"{i}.npy"), restored_latent)
        
    # restored_img = shuffler.shuffle_image(restored_patches)  # (B, C, H, W)
    restored_img = patchfier.merge_patches(restored_patches)  # (B, C, H, W)
    save_image(restored_img ,  os.path.join(output_folder,  f"{image_name}.png"))
"""