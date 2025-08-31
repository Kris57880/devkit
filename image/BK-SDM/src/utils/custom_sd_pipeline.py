import torch
import os
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from .noise_regularization import noise_regularization
from torchvision.utils import save_image
import numpy as np 
import random
noise_regularization_lambda_ac = 20.0
noise_regularization_lambda_kl = 0.065
noise_regularization_num_reg_steps = 4
noise_regularization_num_ac_rolls = 5

def get_fresh_generator(base_generator, offset=0):
    torch.manual_seed(base_generator.initial_seed())
    if base_generator is not None:
        new_gen = torch.Generator(device=base_generator.device)
        new_gen.manual_seed(base_generator.initial_seed() + offset)
        return new_gen
    return None

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

# More detailed range configuration
X_DISTANCE_RANGES = [
    (0.02, (0, 0.02)),
    (0.05, (0, 0.05)),
    (0.1, (0, 0.1)),
    (0.2, (0, 0.2)),
    (0.3, (0, 0.3)),
    (0.4, (0, 0.4)),
    (0.5, (0, 0.5)),
    (0.6, (0, 0.6)),
    (0.8, (0, 0.8)),
    (1.0, (0, 1.0)),
    (float('inf'), (0, 2.0))  # Cases exceeding 1.0
]

Z_DISTANCE_RANGES = [
    (0.05, (0, 0.05)),
    (0.1, (0, 0.1)),
    (0.3, (0, 0.3)),
    (0.5, (0, 0.5)),
    (0.8, (0, 0.8)),
    (1.0, (0, 1.0)),
    (5.0, (0, 5.0)),
    (10.0, (0, 10.0)),
    (15.0, (0, 15.0)),
    (20.0, (0, 20.0)),
    (25.0, (0, 25.0)),
    (35.0, (0, 35.0)),
    (45.0, (0, 45.0)),
    (65.0, (0, 65.0)),
    (float('inf'), (0, 100.0))  # Cases exceeding 60.0
]

def get_plot_ylim_precise(distances, distance_type='x'):
    """
    More precise y-axis range setting
    """
    if not distances:
        return (0, 1)
    
    max_val = max(distances)
    ranges = X_DISTANCE_RANGES if distance_type == 'x' else Z_DISTANCE_RANGES
    
    for threshold, ylim in ranges:
        if max_val <= threshold:
            return ylim
    
    # If none match, return the last range
    return ranges[-1][1]

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    
class StableDiffusionPGDMPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=1, # default no guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        # PGDM-specific
        measurement=None,
        measurement_type = 'image',# latent or image
        operator=None,
        grad_term_weight=1.0,
        start_timestep=0,
        log_dir=None,
    ):
        # 0~6. Same as StableDiffusionPipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)

        num_channels_latents = self.unet.config.in_channels
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # PGDM-specific: initialize from measurement if start_timestep > 0
        if measurement_type == 'latent':
            mea_z0 = measurement
            measurement = self.vae.decode(measurement / self.vae.config.scaling_factor).sample
        else:
            mea_z0 = self.vae.encode(measurement).latent_dist.mean
            mea_z0 = mea_z0 *self.vae.config.scaling_factor
        
        # start the denoising from a specific timestep
        if start_timestep != num_inference_steps:
            timesteps = timesteps[-start_timestep:-1]
            t = timesteps[0]
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(prompt_embeds)            
            latents = alpha_t.sqrt() * mea_z0 + (1 - alpha_t).sqrt() * randn_tensor(mea_z0.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            generator = get_fresh_generator(generator)
        else :
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
            generator = get_fresh_generator(generator)
        distances = []
        image_steps = []
        log_steps = []
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        
        # 7. Denoising loop with PGDM guidance
        for i, t in enumerate(timesteps):
            with torch.enable_grad():
                latents = latents.requires_grad_()
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(latents)
                alpha_s = self.scheduler.alphas_cumprod[t-1].view(-1, 1, 1, 1).to(latents)
                    
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                zt_pred = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                z0_pred = (zt_pred - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

                x0_pred = self.vae.decode(z0_pred/ self.vae.config.scaling_factor).sample
                x0_pred = x0_pred.clamp(-1,1)
                # print(x0_pred.requires_grad, z0_pred.requires_grad, zt_pred.requires_grad, latents.requires_grad)
                image_steps.append(x0_pred.detach())
                # PGDM guidance
                if log_dir is not None:
                    save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_{t}.png"))
                    save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_cur.png"))
                if mea_z0 is not None and operator is not None:
                    n = x0_pred.shape[0]
                    A_x0_pred = operator(x0_pred)
                    A_z0_pred = self.vae.encode(A_x0_pred).latent_dist.mean * self.vae.config.scaling_factor
                    if torch.isnan(A_x0_pred).any():
                        print("operator output is NaN!")
                    if torch.isinf(A_x0_pred).any():
                        print("operator output is INF")
                    mat_z = (mea_z0 - A_z0_pred).reshape(n, -1)
                    mat_z = (mat_z.detach() * z0_pred.reshape(n, -1)).sum()
                    # mat_x = (mea_x0 - A_x0_pred).reshape(n,-1)
                    # mat_x = (mat_x.detach() * x0_pred.reshape(n, -1)).sum()

                    mse_loss = torch.nn.functional.mse_loss(measurement.detach(), A_x0_pred.detach())
                    distances.append(mse_loss.item())
                    log_steps.append(t.item())  

                    # print(f"Step: {t}, distance: {mat_x.item():.4f}")
                    guidance = torch.autograd.grad(mat_z, latents, retain_graph=False)[0].detach()
                    if torch.isnan(guidance).any():
                        print("guidance is NaN! Zeroing out.")
                        guidance = torch.zeros_like(latents)
                    # PGDM update
                    c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * eta
                    c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
                    coeff = alpha_t.sqrt() * grad_term_weight
                    torch.manual_seed(generator.initial_seed())
                    latents = alpha_s.sqrt() * z0_pred.detach() + c1 * torch.randn_like(latents, device=device) + c2 * noise_pred.detach() + guidance * coeff
                    latents = latents.detach()
                else:
                    latents = zt_pred.detach()
                
                del noise_pred, z0_pred, zt_pred, x0_pred#, mat, mat_x, guidance
                torch.cuda.empty_cache()
                
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            
        # Plot distance vs. steps if PGDM
        if distances and log_dir is not None:
            plt.figure()
            plt.plot(log_steps, distances)
            plt.xlabel("Step")
            plt.ylabel("Distance")
            plt.gca().invert_xaxis()
            plt.title("Distance vs. Steps")
            plt.savefig(os.path.join(log_dir, "distance_vs_steps.png"))
            plt.close()

        # ...existing code for decode, safety checker, output...
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

class StableDiffusionMPGDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=1, # default no guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        # MPGD-specific
        measurement=None,
        measurement_type = 'image',# latent or image
        operator=None,
        grad_term_weight=1.0,
        start_timestep=0,
        log_dir=None,
    ):
        set_all_seeds(generator.initial_seed())
        generator = get_fresh_generator(generator)
        # 0~6. Same as StableDiffusionPipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)

        num_channels_latents = self.unet.config.in_channels
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # initialize from measurement 
        if measurement_type == 'latent':
            mea_z0 = measurement
            measurement = self.vae.decode(measurement / self.vae.config.scaling_factor).sample
        else:
            mea_z0 = self.vae.encode(measurement).latent_dist.mean
            mea_z0 = mea_z0 *self.vae.config.scaling_factor
        # start the denoising from a specific timestep
        if start_timestep != num_inference_steps:
            timesteps = timesteps[-start_timestep:-1]
            t = timesteps[0]
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(prompt_embeds)            
            latents = alpha_t.sqrt() * mea_z0 + (1 - alpha_t).sqrt() * randn_tensor(mea_z0.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        else :
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
        distances = []
        log_steps = []
        image_steps = []
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        # 7. Denoising loop with MPGD guidance
        for i, t in enumerate(timesteps):
            with torch.enable_grad():
                latents = latents.requires_grad_()
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(latents)
                alpha_s = self.scheduler.alphas_cumprod[t-1].view(-1, 1, 1, 1).to(latents)
                    
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                zt_pred = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                z0_pred = (zt_pred - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

                x0_pred = self.vae.decode(z0_pred/ self.vae.config.scaling_factor).sample
                x0_pred = x0_pred.clamp(-1,1)
                image_steps.append(x0_pred.detach())
                # MPGD guidance
                if log_dir is not None:
                    save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_{t}.png"))
                    save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_cur.png"))
                if measurement is not None and operator is not None:
                    n = x0_pred.shape[0]
                    A_x0_pred = operator(x0_pred)
                    # op_out_latent = self.vae.encode(op_out).latent_dist.sample()*self.vae.config.scaling_factor
                    if torch.isnan(A_x0_pred).any():
                        print("operator output is NaN!")
                    if torch.isinf(A_x0_pred).any():
                        print("operator output is INF")
                    # mat_z = (latent_measurement - op_out_latent).reshape(n, -1)
                    # mat_z = (mat_z.detach() * z0_pred.reshape(n, -1)).sum()
                    mat_x = (measurement - A_x0_pred)
                    norm = torch.linalg.norm(mat_x)
                

                    mse_loss = torch.nn.functional.mse_loss(measurement.detach(), A_x0_pred.detach())
                    distances.append(mse_loss.item())
                    log_steps.append(t.item())

                    guidance  = torch.autograd.grad(outputs=norm, inputs=z0_pred, retain_graph=False)[0]
        
                    if torch.isnan(guidance).any():
                        print("guidance is NaN! Zeroing out.")
                        guidance = torch.zeros_like(latents)
                    
                    z0_pred = z0_pred.detach() - guidance * grad_term_weight
                    latents = alpha_s.sqrt() * z0_pred + (1 - alpha_s).sqrt() * noise_pred.detach() #DDIM
                else:
                    latents = zt_pred.detach()
                
                del noise_pred, z0_pred, zt_pred, x0_pred, mat_x, norm, guidance, A_x0_pred
                torch.cuda.empty_cache()
                
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
            
        # Plot distance vs. steps if PGDM
        if distances and log_dir is not None:
            plt.figure()
            plt.plot(log_steps, distances)
            plt.xlabel("Step")
            plt.ylabel("Distance")
            plt.gca().invert_xaxis()
            plt.title("Distance vs. Steps")
            plt.savefig(os.path.join(log_dir, "distance_vs_steps.png"))
            plt.close()

        # ...existing code for decode, safety checker, output...
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

class StableDiffusionNormalPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=1, # default no guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        # MPGD-specific
        measurement=None,
        start_timestep=0,
        log_dir=None,
    ):
        set_all_seeds(generator.initial_seed())
        generator = get_fresh_generator(generator)
        # 0~6. Same as StableDiffusionPipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)

        num_channels_latents = self.unet.config.in_channels
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # initialize from measurement if start_timestep > 0
        mea_z0 = self.vae.encode(measurement).latent_dist.mean
        mea_z0 = mea_z0 *self.vae.config.scaling_factor
        # start the denoising from a specific timestep
        if start_timestep != num_inference_steps:
            timesteps = timesteps[-start_timestep:-1]
            t = timesteps[0]
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(prompt_embeds)            
            latents = alpha_t.sqrt() * mea_z0 + (1 - alpha_t).sqrt() * randn_tensor(mea_z0.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        else :
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
        
        image_steps = []
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        # 7. Denoising loop with MPGD guidance
        for i, t in enumerate(timesteps):
            latents = latents.requires_grad_()
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(latents)
            alpha_s = self.scheduler.alphas_cumprod[t-1].view(-1, 1, 1, 1).to(latents)
                
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            zt_pred = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            z0_pred = (zt_pred - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            x0_pred = self.vae.decode(z0_pred/ self.vae.config.scaling_factor).sample
            x0_pred = x0_pred.clamp(-1,1)
            image_steps.append(x0_pred.detach())
            # MPGD guidance
            if log_dir is not None:
                save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_{t}.png"))
                save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_cur.png"))
            
            latents = zt_pred.detach()
            
            del noise_pred, z0_pred, zt_pred, x0_pred
            torch.cuda.empty_cache()
            
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
        
        # ...existing code for decode, safety checker, output...
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class StableDiffusionNCPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=1, # default no guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        # nc-specific
        measurement=None,
        operator=None,
        log_dir=None,
        start_timestep=0,
        cali_N=10,
        do_noise_regularization=True,
        N_start_reg=7,
        denoise_after_nc=True,
    ):
        set_all_seeds(generator.initial_seed())
        generator = get_fresh_generator(generator)

        # 0~6. Same as StableDiffusionPipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)

        num_channels_latents = self.unet.config.in_channels
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        #initialize from measurement if start_timestep > 0
        mea_z0 = self.vae.encode(measurement).latent_dist.mean*self.vae.config.scaling_factor
        # start the denoising from a specific timestep
        
        x_distances = []
        z_distances = []
        image_steps = []
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        timesteps = timesteps[-start_timestep:-1]
        t = timesteps[0]
        alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(prompt_embeds)

        # 7. Noise Calibration
        best_z0_pred = None
        min_x0_loss = float('inf')

        # generator = get_fresh_generator(generator)
        noise_t = randn_tensor(mea_z0.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        noise_t_ori = noise_t
        
        # Noise Calibration loop
        for i  in range(cali_N):
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(prompt_embeds)            
            latents = alpha_t.sqrt() * mea_z0 + (1 - alpha_t).sqrt() * noise_t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            zt_pred = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            z0_pred = (zt_pred - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            x0_pred = self.vae.decode(z0_pred/ self.vae.config.scaling_factor).sample
            x0_pred = x0_pred.clamp(-1,1)
            image_steps.append(x0_pred.detach())

            # Noise Calibration 
            if log_dir is not None:
                save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_{t}_iter_{i}.png"))
                save_image((x0_pred + 1.0) / 2.0, os.path.join(f"nc_x0_cur.png"))
            n = x0_pred.shape[0]
            A_x0_pred = operator(x0_pred)
            # A_z0_pred = retrieve_latents(self.vae.encode(A_x0_pred), generator=generator)*self.vae.config.scaling_factor
            
            A_z0_pred = self.vae.encode(A_x0_pred).latent_dist.mean*self.vae.config.scaling_factor
            # z0_pred_enc = self.vae.encode(x0_pred).latent_dist.mean*self.vae.config.scaling_factor #v2 
            if torch.isnan(A_x0_pred).any():
                print("operator output is NaN!")
            if torch.isinf(A_x0_pred).any():
                print("operator output is INF")
            z_difference = z0_pred - A_z0_pred 
            # z_difference =  z0_pred_enc-A_z0_pred #v2 
            z_distance = torch.nn.functional.mse_loss(A_z0_pred.detach(), mea_z0.detach())
            x_distance = torch.nn.functional.mse_loss(A_x0_pred.detach(), measurement.detach())
            x_distances.append(x_distance.item())
            z_distances.append(z_distance.item())
            if x_distance < min_x0_loss:
                min_x0_loss = x_distance
                best_z0_pred = z0_pred.detach()
        
            noise_t = noise_pred + (alpha_t.sqrt() / (1 - alpha_t).sqrt())* z_difference 
            # regularize noise
            if do_noise_regularization and i >= N_start_reg:
                # print('noise_regularization at step ', i)
                with torch.enable_grad():
                    noise_t = noise_t.detach()
                    noise_t_ori = noise_t_ori.detach()
                    noise_t = noise_regularization(noise_t, noise_t_ori, lambda_kl=noise_regularization_lambda_kl, lambda_ac=noise_regularization_lambda_ac, num_reg_steps=noise_regularization_num_reg_steps, num_ac_rolls=noise_regularization_num_ac_rolls, generator=generator)

            noise_t = noise_t.detach()
            
        latents = best_z0_pred.detach()
        del noise_pred, z0_pred, zt_pred, x0_pred#, mat, mat_x, guidance
        torch.cuda.empty_cache()
        
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

        # denoising loop using calibrated noise_t 
        if denoise_after_nc:
            noise_t = noise_t.detach()
            latents = alpha_t.sqrt() * mea_z0 + (1 - alpha_t).sqrt() * noise_t 
            
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.requires_grad_()
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(latents)
                        
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    zt_pred = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    latents = zt_pred.detach()
                    z0_pred = (zt_pred - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

                    x0_pred = self.vae.decode(z0_pred/ self.vae.config.scaling_factor).sample
                    x0_pred = x0_pred.clamp(-1,1)
                    # print(x0_pred.requires_grad, z0_pred.requires_grad, zt_pred.requires_grad, latents.requires_grad)
                    image_steps.append(x0_pred.detach())
                
                    if log_dir is not None:
                        save_image((x0_pred + 1.0) / 2.0, os.path.join(log_dir, f"x0_{t}_iter_{i}.png"))
                        save_image((x0_pred + 1.0) / 2.0, os.path.join(f"x0_cur.png"))

                    del noise_pred , zt_pred, x0_pred, z0_pred #, mat, mat_x, guidance
                    torch.cuda.empty_cache()
                    
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        # Plot distance for NC
        if (x_distances or z_distances) and log_dir is not None:
            plt.figure(figsize=(10, 4))
            
            # Calculate respective y-axis ranges
            x_ylim = get_plot_ylim_precise(x_distances, 'x')
            z_ylim = get_plot_ylim_precise(z_distances, 'z')
            
            plt.subplot(1, 2, 1)
            plt.plot(range(len(x_distances)), x_distances, 'b-', linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("X Distance")
            plt.title("X Distance vs. Steps")
            plt.ylim(x_ylim)  # Set fixed range
            plt.grid(True, alpha=0.3)
            
            # Add max and min value annotations
            if x_distances:
                max_idx = x_distances.index(max(x_distances))
                min_idx = x_distances.index(min(x_distances))
                plt.annotate(f'Max: {max(x_distances):.4f}', 
                           xy=(max_idx, max(x_distances)), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                plt.annotate(f'Min: {min(x_distances):.4f}', 
                           xy=(min_idx, min(x_distances)), 
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.subplot(1, 2, 2)
            plt.plot(range(len(z_distances)), z_distances, 'r-', linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("Z Distance")
            plt.title("Z Distance vs. Steps")
            plt.ylim(z_ylim)  # Set fixed range
            plt.grid(True, alpha=0.3)
            
            # Add max and min value annotations
            if z_distances:
                max_idx = z_distances.index(max(z_distances))
                min_idx = z_distances.index(min(z_distances))
                plt.annotate(f'Max: {max(z_distances):.4f}', 
                           xy=(max_idx, max(z_distances)), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                plt.annotate(f'Min: {min(z_distances):.4f}', 
                           xy=(min_idx, min(z_distances)), 
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "distance_vs_steps.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            
        # ...existing code for decode, safety checker, output...
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), StableDiffusionPipelineOutput(latents, nsfw_content_detected=has_nsfw_concept)



class StableDiffusionVAEOnlyPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=1, # default no guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        measurement = None # VAE-only specific
    ):
        set_all_seeds(generator.initial_seed())
        generator = get_fresh_generator(generator)

        # 0~6. Same as StableDiffusionPipeline
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )[0]
        latents = self.vae.encode(measurement).latent_dist.mean*self.vae.config.scaling_factor

        # ...existing code for decode, safety checker, output...
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":
    from degradations import VVC, MS_ILLM
    from PIL import Image
    import numpy as np
    import torch
    import os
    device = "cuda"

    # Load and preprocess image
    src = Image.open("../../sample.png")
    src = src.resize((512, 512), Image.BICUBIC)
    src = torch.from_numpy(np.array(src, dtype=np.float16)).permute(2, 0, 1)[None]
    src = (src / 127.5) - 1.0
    src = src.to(device)

    # Set up  operator
    
    operator = MS_ILLM(qf=1).H  
    measurement = operator(src)

    save_image((src + 1.0) / 2.0, "pgdm_src.png")
    save_image((measurement + 1.0) / 2.0, "pgdm_mea.png")
    model_path = "nota-ai/bk-sdm-small"
    # model_path = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionNCPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    h, w = measurement.shape[2], measurement.shape[3]
    pipe = pipe.to(device)
    prompt = "a high resolution photograph, 4K, 8K, detailed, realistic, high quality, sharp focus, intricate details, hyper-realistic, ultra-detailed"
    generator = torch.Generator(device=device).manual_seed(1024)
    image = pipe(
            prompt=prompt, 
            height=h,
            width=w,        
            generator=generator, 
            measurement=measurement,
            operator=operator,
            guidance_scale = 0, # unconditional
            grad_term_weight=0.01, 
            start_timestep =5, 
            log_dir = 'log'
            ).images[0]
    image.save("sd_pgdm_generated_image.png")