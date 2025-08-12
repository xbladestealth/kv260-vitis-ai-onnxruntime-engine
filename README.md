# Vitis AI Docker Environment

## Build Docker Image for Vitis AI 3.5.1

```bash
cd Downloads/workspace/
git clone https://github.com/xbladestealth/Vitis-AI -b 3.5.1
cd Vitis-AI/docker/
./docker_build.sh -t gpu -f pytorch
```

```bash
docker images
REPOSITORY                        TAG                   IMAGE ID       CREATED          SIZE
xilinx/vitis-ai-pytorch-gpu       3.5.1.001-ad2b6ded7   927a3d10b650   46 minutes ago   33.3GB
xiinx/vitis-ai-gpu-pytorch-base   latest                03c687f7c06b   2 weeks ago      11.2GB
```

## Run Docker Container

```bash
cd Vitis-AI/
bash docker_run.sh xilinx/vitis-ai-pytorch-gpu:3.5.1.001-ad2b6ded7
conda activate vitis-ai-pytorch
```

## Install Vitis AI ONNXRuntime Engine

Install VOE n the docker container

```bash
pip show vai_q_onnx
WARNING: Package(s) not found: vai_q_onnx
```

```bash
cd src/vai_quantizer/vai_q_onnx/
python -m pip install -r requirements.txt
chmod +x build.sh
./build.sh
pip install pkgs/*.whl
```

```bash
pip show vai_q_onnx
Name: vai-q-onnx
Version: 1.14.0
Summary: Xilinx Vitis AI Quantizer for ONNX. It is customized based on [Quantization Tool](https://github.com/microsoft/onnxruntime/tree/rel-1.14.0/onnxruntime/python/tools/quantization).
Home-page: 
Author: Xiao Sheng
Author-email: kylexiao@xilinx.com
License: Apache 2.0
Location: /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.8/site-packages
Requires: numpy, onnx, onnxruntime, onnxruntime-extensions, protobuf
Required-by: 
```

## VOE Quantization Workflow

```bash
cd /workspace/
git clone https://github.com/xbladestealth/kv260-vitis-ai-onnxruntime-engine.git
cd kv260-vitis-ai-onnxruntime-engine/scripts/mnist/
```

### MNIST CNN Model

Prepare it by yourself
```bash
python prepare_model_data.py
```

or download by:

```bash
wget https://huggingface.co/onnxmodelzoo/legacy_models/resolve/main/validated/vision/classification/mnist/model/mnist-12.onnx -O mnist_cnn.onnx
```

Check out the model by running:
```bash
python run_inference_mnist.py --provider cpu --model mnist_cnn.onnx --input input.npy
```

### Preprocess

```bash
python -m onnxruntime.quantization.preprocess --input mnist_cnn.onnx --output mnist_cnn_preproc.onnx
```

### Quantize

```bash
python quantize_static.py
```

# Build Vitis Platform and DPU Application

Create custom Vitis platform by following the instructions in the link below:  
- https://github.com/Xilinx/Vitis-Tutorials/blob/2023.1/Vitis_Platform_Creation/Design_Tutorials/01-Edge-KV260/step1.md

Instead of following the Step 3, follow the instructions to build DPU application in link below:  
- https://qiita.com/basaro_k/items/dc439ffbc3ea3aed5eb2
- https://misoji-engineer.com/archives/vitis-ai-3-5-kv260-yolox.html

After both the vitis platform and DPU application, prepare files to copy to your sd card:

```bash
cd Downloads/workspace/kv260/
mkdir -p sd_card/dpu/
cd sd_card/
cp kv260_vitis_platform/dtg_output/pl.dtbo dpu/
cp kv260_vitis_application/dpu_system_hw_link/Hardware/dpu.xclbin dpu/dpu.bin
cp kv260_vitis_application/dpu_system_hw_link/Hardware/dpu.build/link/vivado/vpl/prj/prj.gen/sources_1/bd/design_1/ip/design_1_DPUCZDX8G_1_0/arch.json dpu/
touch dpu/shell.json
touch dpu/vart.conf
```

Edit shell.json:
```json
shell_type	"XRT_FLAT"
num_slots	"1"
```

Edit vart.conf:
```bash
firmware: /lib/firmware/xilinx/dpu/dpu.bin
```

Note that `arch.json` is needed only when you compile a quantized model and export a `*.xmodel`, but we just copy it to `dpu/` so that we don't have to look for directories to find it.

# Run MNIST CNN with DPU on KV260

## Flash SD Card (PC)
Flash KV260 SD image (`xilinx-v2023.1_kv260_sdimage.zip`) to your sd card using `balenaEtcher`.

## Serial Connection (PC)

Attach to KV260 for the first time:
```bash
sudo screen /dev/ttyUSB1 115200
```

## Setup SSH (KV260)

Configure the IP address for ssh/scp on KV260:
```bash
sudo ifconfig eth1 192.168.0.11
ping 192.168.0.10
```

where `192.168.0.10` is the ip address of your PC.

## Copy Files (PC)

```bash
cd Downloads/workspace/kv260/sd_card/
scp -r dpu/ petalinux@192.168.0.11:~/
```

```bash
cd Downloads/workspace/downloads/
scp vitis_ai_2023.1-r3.5.0.tar.gz voe-0.1.0-py3-none-any.whl onnxruntime_vitisai-1.16.0-py3-none-any.whl petalinux@192.168.0.11:./
```

```bash
cd Downloads/workspace/Vitis-AI/kv260-vitis-ai-onnxruntime-engine/scripts/
scp -r mnist/ petalinux@192.168.0.11:~/
```

## Install Dependencies (KV260)

```bash
sudo dnf install zocl-202310.2.15.0-r0.0.zynqmp_generic
sudo dnf install xrt-202310.2.15.0-r0.0.cortexa72_cortexa53
sudo dnf install packagegroup-petalinux-opencv
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-3.5.0.tar.gz -O vitis-ai-runtime-3.5.0.tar.gz
tar -xzvf vitis-ai-runtime-3.5.0.tar.gz
cd vitis-ai-runtime-3.5.0/2023.1/aarch64/centos/
sudo bash ./setup.sh
```

## Install Vitis AI ONNXRuntime Engine (KV260)
```bash
cd ~
sudo tar -xzvf vitis_ai_2023.1-r3.5.0.tar.gz -C /
pip3 install voe*.whl
pip3 install onnxruntime_vitisai*.whl
```

## Startup DPU (KV260)

```bash
cd ~/dpu/
sudo mkdir /lib/firmware/xilinx/dpu
sudo cp dpu.bin pl.dtbo shell.json /lib/firmware/xilinx/dpu/
sudo cp vart.conf /etc/
```

Load DPU:
```bash
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp dpu
```

## Run MNIST (KV260)

```bash
cd ~/mnist/
python run_inference_mnist.py --provider dpu --model mnist_cnn_quantized.onnx --input input.npy
```
