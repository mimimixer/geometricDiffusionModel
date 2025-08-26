import torch
from diffusers import DDPMScheduler
from diffusers.utils import load_image
from sympy import plot_implicit

from diffusion2 import timesteps
from diffusion2a import noisy_image

max_timesteps = 10 #number of steps to turn image to noise

#initialize scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=max_timesteps,
                                beta_start=0.0001,
                                beta_step=0.02
                                )

#load image
image = torch.tensor(load_image())  #video says load_img() but where is it?
img_shape = image.shape

timesteps = torch.arange(1, max_timesteps)
noise = torch.randn(img_shape)  # Returns a tensor filled with random numbers from a normal distribution
                                # with mean `0` and variance `1` (also called the standard normal distribution).

# add noise to image
noisy_images = noise_scheduler.add_noise(image, noise, timesteps).numpy()

# plot
plot_images(timesteps, noisy_images)

