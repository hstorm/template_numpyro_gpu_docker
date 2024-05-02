# This repository is a template for running NumPyro on GPU using JAX in a vs code devcontainer

The docker image contains preinstalled  CUDA Toolkit and NVIDIA cuDNN. This should make it easier to run NumPyro on GPU and avoid one shared CUDA installation.  

1. Either use this repository as a stating point or copy the `.devcontainer` folder, the Dockerfile, the  `requirements.txt` and the `text.py` file to your project.

2. Adjust the `requirements.txt` file to your needs. Note that this template does not use conda or pipenv. Instead plain pip is used to install packages. When you `pip install`  additional packages later, remember to update the `requirements.txt` file accordingly, otherwise these packages will be lost when the container is rebuild.

3. Open the project in VS Code and hit `Ctrl+Shift+P` and select `Dev Containers: Reopen in Container`.
Note: For this to work you user needs to be in the docker user group (`sudo usermod -a -G docker USERNAME`)

# Notes
Choice other docker image 
from https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/index.html

Here `nvcr.io/nvidia/jax:23.10-py3` is used.

This is based on jaxlib==0.4.17 and works with numypro Version 0.14.0



### [Alternative for testing puposes] Manually build docker image
- Build docker image: `docker build -t numpyro_gpu_template:0.1 .`

- To Start container `docker run --gpus all -it --rm numpyro_gpu_template:0.1`