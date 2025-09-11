# Generating geometrical shapes with AI models
### Exploring the latent space of VAE, GAN and Diffusion models

* create your dataset with: generateGeometricShapes.py

- chose your model, type in the folder with your dataset, and train: 
    - for VAE with cnn use cnnVAEtrain.py
    - for GAN use GAN.py
    - for diffusion use diffusion5.py
  it will not only train but also visualize the latent space and infer some samples after training (VAE) or document inference after each 10% of total epochs (GAN and diffusion). 

Other possibilities to combine fc-VAE with one layer in /VAE/fcVAEmodel.py or with more layers in VAEmodel.py are also there and can be trained with VAEtrain.py and inference with VAEgenerator.py and visualisation with visualizeVAE.py.

There is also the possibility to use PCA on your dataset with PCA.py. This will automatically generate some eigenfaces (in this case eigenshapes) and a recreation of one random training-shape. 
