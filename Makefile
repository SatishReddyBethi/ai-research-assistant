.phony: 

# Source: https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=linux%2Fwsl2&package=pip 
install_torch_with_intel_xpu_support:=
	python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/xpu
	# Optional: python -m pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
	python -m pip install intel-extension-for-pytorch==2.8.10+xpu oneccl_bind_pt==2.8.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
	#pip install mkl mkl-include

uninstall_torch_with_intel_xpu_support:
	pip uninstall torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt

# Needs to be run using "sudo"
# Install Intel Level Zero loader library (libze_loader.so.1).
# Source: https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=linux%2Fwsl2&package=pip
# Source: https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest
# Source: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
install_intel_xpu_dependencies_ubuntu_24_04:
	# Refresh the local package index and install the package for managing software repositories.
	sudo apt-get update
	sudo apt-get install -y software-properties-common

	# Add the intel-graphics PPA.
	sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
	
	# Install the compute-related packages.
	sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc

	# Install the media-related packages.
	sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo

	# Install libze-dev and intel-ocloc additionally for PyTorch
	sudo apt-get install -y libze-dev intel-ocloc

	# (Optional) Install 'libze-intel-gpu-raytracing' to enable hardware ray tracing support
	# sudo apt-get install -y libze-intel-gpu-raytracing

	# You may need to have gomp package in your system
	sudo apt install libgomp1

	# Installing this should not be neccessary.
	# The 'torch' and 'intel-extension-for-pytorch' should be self contained and include this toolkit
	# make install_intel_oneapi_base_toolkit_ubuntu

# Source: https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=linux%2Fwsl2&package=pip
# Source: https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-22.04
install_intel_xpu_dependencies_ubuntu_22_04:
	# Install the Intel graphics GPG public key.
	wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
	sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

	# Configure the repositories.intel.com package repository.
	echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  	sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

	# Update the package repository metadata.
	sudo apt update

	# Install the compute-related packages.
	sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo

	# Install libze-dev and intel-ocloc additionally for PyTorch
	sudo apt-get install -y libze-dev intel-ocloc

	# (Optional) Install 'intel-level-zero-gpu-raytracing' to enable hardware ray tracing support
	# sudo apt-get install -y intel-level-zero-gpu-raytracing

	# You may need to have gomp package in your system
	sudo apt install libgomp1

	# Installing this should not be neccessary.
	# The 'torch' and 'intel-extension-for-pytorch' should be self contained and include this toolkit
	# make install_intel_oneapi_base_toolkit_ubuntu

install_intel_oneapi_base_toolkit_ubuntu:	
	# Download the key to system keyring
	sudo apt update
	sudo apt install -y gpg-agent wget
	wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
	| gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

	# add signed entry to apt sources and configure the APT client to use Intel repository:
	echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
	sudo tee /etc/apt/sources.list.d/oneAPI.list
	sudo apt update

	# Install the toolkit
	sudo apt install intel-oneapi-base-toolkit

	# Configure the System After Toolkit Installation
	sudo apt update
	sudo apt -y install cmake pkg-config build-essential

verify_intel_xpu_ubuntu_dependencies:
	# To verify that the kernel and compute drivers are installed and functional, run 'clinfo'
	clinfo | grep "Device Name"

	# You should see the Intel graphics product device names listed.
	# If they do not appear, ensure you have permissions to access /dev/dri/renderD*.
	# This typically requires your user to be in the render group
	# sudo gpasswd -a ${USER} render
	# newgrp render

	# OR, you can just run 'clinfo' as root

create_python_venv:
	python3 -m venv aia_venv
	source aia_venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	# Only needed if using Intel XPU (GPU)
	# make install_torch_with_intel_xpu_support

activate_python_venv:
	source aia_venv/bin/activate

deactivate_python_venv:
	deactivate