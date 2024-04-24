# This repository is a template for running NumPyro on GPU using JAX in a vs code devcontainer

1. Either that this repository as a stating point or copy the `.devcontainer` folder, the Dockerfile, the  `requirements.txt` and the `text.py` file to your project.

2. Adjust the `requirements.txt` file to your needs. Note that this template does using conda or pipenv. Instead plain pip is used to install the packages. You can also install pip later on to install additional packages, but remember to update the `requirements.txt` file, otherwise those install will be lost when the container is rebuild.

3. Open the project in VS Code and hit `Ctrl+Shift+P` and select `Remote-Containers: Reopen in Container`

# Notes
Choice other docker image 
from https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/index.html

Here `nvcr.io/nvidia/jax:23.10-py3` is used.

This is based on jaxlib==0.4.17 and works with numypro Version 0.14.0



### [Alternative for testing puposes] Manually build docker image
- Build docker image: `docker build -t numpyro_gpu_template:0.1 .`

- To Start container `docker run --gpus all -it --rm numpyro_gpu_template:0.1`