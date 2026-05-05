# This repository is a template for running NumPyro on GPU using JAX in a vs code devcontainer

__Background Motivation why Docker might be useful__: https://docs.google.com/presentation/d/1B4TgoT1U4Ez6MCB8vGfiBcZ3lI2D0QSRSAo4Lvv80C4/edit?usp=sharing 


The repository allows to create VS Code development container starting from an docker image contains that has CUDA Toolkit, NVIDIA cuDNN and JAX preinstalled. This should make it easier to run NumPyro on GPU and avoid one shared CUDA installation.  

1. Either use this repository as a stating point or copy the `.devcontainer` folder, the Dockerfile, the  `requirements.txt` and the `text.py` file to your project.

2. Adjust the `requirements.txt` file to your needs. Note that this template does not use conda or pipenv. Instead plain pip is used to install packages. When you `pip install`  additional packages later, remember to update the `requirements.txt` file accordingly, otherwise these packages will be lost when the container is rebuild.

3. Open the project in VS Code and hit `Ctrl+Shift+P` and select `Dev Containers: Reopen in Container`.
Note: For this to work you user needs to be in the docker user group (`sudo usermod -a -G docker USERNAME`)

4. Test the installation by running `python src/test_numpyro.py` or `python src/test_gpjax_numpyro.py`. You should see that the code is running on GPU. For this connect to the server used SSH (without an development container) and run `watch -n 1 nvidia-smi` to check that the GPU is used.

(5. optional: To have a correct name for git commits set your name and email in the container: `git config --global user.name "Your Name"` and `git config --global user.email "your.email@example.com"`)

## Other container as base image
In the `Dockerfile` you can adjust the base image to your needs. The base image should have CUDA Toolkit, NVIDIA cuDNN and JAX preinstalled.


Here `nvcr.io/nvidia/jax:26.04-py3` is used. This is provides jax==0.9.2, and works with numpyro==0.21.0 (and gpjax==0.14.0)

For older numpyro versions a working base image is `nvcr.io/nvidia/jax:23.10-py3`. This works with numypro==0.14.0

Choice other docker image 
from https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/index.html

## Notes when using this without a GPU (e.g. when running a container locally on laptop)
In this case adjust ```.devcontainer/devcontainer.json``` replacing line 

    "runArgs": [ "--gpus=all", "-it", "--rm"] with

with 

    "runArgs": [  "-it", "--rm"]

### [Alternative for testing puposes] Manually build docker image
- Build docker image: `docker build -t numpyro_gpu_template:0.1 .`

- To Start container `docker run --gpus all -it --rm numpyro_gpu_template:0.1`