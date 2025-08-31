# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import pi
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from diffusers import DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from torchvision.utils import save_image
import os 

class PGDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for Pseudoinverse Guided Diffusion Model.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        measurement: torch.Tensor,
        operator: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        grad_term_weight: float = 1.0,
        eta: float = 1.0,
        start_timestep: int = 0,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Args:
            measurement: The degraded image tensor.
            operator: The H function, e.g., VVC(...).H.
            loss_fn: Loss function between measurement and operator(x).
        """
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        if start_timestep != num_inference_steps:
            noise = randn_tensor(measurement.shape, generator=generator, device=self.device)
            alpha_cumprod = self.scheduler.alphas_cumprod[timesteps[-start_timestep]].view(-1, 1, 1, 1).to(self.device)
            image = alpha_cumprod.sqrt() * measurement + (1 - alpha_cumprod).sqrt() * noise
            image = image.to(self.device)
            timesteps = timesteps[-start_timestep:-1]
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        distances = []
        for t in self.progress_bar(timesteps):
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1).to(image.device)
            alpha_s = self.scheduler.alphas_cumprod[t-1].view(-1, 1, 1, 1).to(image.device)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            coeff = alpha_t.sqrt() * grad_term_weight
            
            with torch.enable_grad():
                image = image.requires_grad_()
                model_output = self.unet(image, t).sample
                scheduler_out = self.scheduler.step(model_output, t, image, generator=generator)
                image_pred, origi_pred = scheduler_out.prev_sample, scheduler_out.pred_original_sample
                
                save_image((model_output[:,:3,:,:] + 1.0) / 2.0, os.path.join("./log", f"model_output-0_{t}.png"))
                save_image((model_output[:,3:,:,:] + 1.0) / 2.0, os.path.join("./log", f"model_output-3_{t}.png"))
                save_image((image_pred + 1.0) / 2.0, os.path.join("./log", f"xt_{t}.png"))
                save_image((origi_pred + 1.0) / 2.0, os.path.join("./log", f"x0_{t}.png"))
                save_image((origi_pred + 1.0) / 2.0, os.path.join("./log", f"x0_cur.png"))
                n = image_pred.shape[0]
                mat = (measurement- operator(origi_pred)).reshape(n, -1)
                mat_x = (mat.detach() * origi_pred.reshape(n,-1)).sum()
                print("distance: {0:.4f}".format(mat_x.item()))
                distances.append(mat_x.item())
                guidance = torch.autograd.grad(mat_x, image, retain_graph=False)[0].detach()
                
                image = alpha_s.sqrt() * origi_pred.detach() + c1 * torch.randn_like(image, device=image.device) + c2 * model_output.detach() + guidance * coeff
                image = image.detach()
                del image_pred, origi_pred, mat, mat_x, guidance
                torch.cuda.empty_cache()

        # Plot distance vs. steps
        plt.figure()
        plt.plot(range(len(distances)), distances)
        plt.xlabel("Step")
        plt.ylabel("Distance")
        plt.title("Distance vs. Steps")
        plt.savefig(os.path.join("./log","distance_vs_steps.png"))
        plt.close()

        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
        
class DPSPipeline(DiffusionPipeline):
    r"""
    Pipeline for Diffusion Posterior Sampling.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        measurement: torch.Tensor,
        operator: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        zeta: float = 0.3,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            measurement (`torch.Tensor`, *required*):
                A 'torch.Tensor', the corrupted image
            operator (`torch.nn.Module`, *required*):
                A 'torch.nn.Module', the operator generating the corrupted image
            loss_fn (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *required*):
                A 'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]', the loss function used
                between the measurements, for most of the cases using RMSE is fine.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            with torch.enable_grad():
                # 1. predict noise model_output
                image = image.requires_grad_()
                model_output = self.unet(image, t).sample

                # 2. compute previous image x'_{t-1} and original prediction x0_{t}
                scheduler_out = self.scheduler.step(model_output, t, image, generator=generator)
                image_pred, origi_pred = scheduler_out.prev_sample, scheduler_out.pred_original_sample

                # 3. compute y'_t = f(x0_{t})
                measurement_pred = operator(origi_pred)

                # 4. compute loss = d(y, y'_t-1)
                loss = loss_fn(measurement, measurement_pred)
                loss.backward()

                print("distance: {0:.4f}".format(loss.item()))

                with torch.no_grad():
                    image_pred = image_pred - zeta * image.grad
                    image = image_pred.detach()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


if __name__ == "__main__":
    from degradations import VVC
    from diffusers import DDPMScheduler, UNet2DModel
    from torchvision.utils import save_image
    from PIL import Image
    import numpy as np
    import torch
    import os

    def RMSELoss(yhat, y):
        return torch.sqrt(torch.sum((yhat - y) ** 2))

    # Load and preprocess image
    src = Image.open("../../sample.png")
    src = torch.from_numpy(np.array(src, dtype=np.float32)).permute(2, 0, 1)[None]
    src = (src / 127.5) - 1.0
    src = src.to("cuda")

    # Set up VVC operator
    vvc_operator = VVC(qp=37, vvc_root='/home/pc3400/kris/VVCSoftware_VTM').H  # 修改路徑

    measurement = vvc_operator(src)

    scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(1000)
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to("cuda")

    save_image((src + 1.0) / 2.0, "pgdm_src.png")
    save_image((measurement + 1.0) / 2.0, "pgdm_mea.png")

    pgdm_pipe = PGDMPipeline(model, scheduler)
    image = pgdm_pipe(
        measurement=measurement,
        operator=vvc_operator,
        grad_term_weight=0.5,
        eta=1.0,
        num_inference_steps=1000,
        start_timestep=50,
    ).images[0]
    image.save("pgdm_generated_image.png")