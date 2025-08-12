# Vitis AI Environment

## Build Docker Image for Vitis AI 3.5.1

```bash
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

# Install Vitis AI ONNXRuntime Engine

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

# VOE Workflow Demo with MNIST

```bash
cd /workspace/
git clone https://github.com/xbladestealth/kv260-vitis-ai-onnxruntime-engine.git
cd kv260-vitis-ai-onnxruntime-engine/scripts/mnist/
```

## Base Model

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
python run_inference_mnist.py
```

## Preprocess

```bash
python xxx
```

## Quantize

```bash
python yyy
```


## 


<!-- ## Download
Download the ONNX model file from ONNX github repository:
```bash
mkdir -p models/mnist
cd models/mnist/
wget https://huggingface.co/onnxmodelzoo/legacy_models/resolve/main/validated/vision/classification/mnist/model/mnist-12.onnx -O mnist.onnx
``` -->

<!-- ## Preprocess
Pre-process prepares a float32 model for quantization.
```bash
python -m onnxruntime.quantization.preprocess --input mnist.onnx --output mnist_preproc.onnx
```

Otherwise you might get this warning:
```bash
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
```

## Quantize
Create a Python script `quantize_mnist.py` to quantize the model:
```bash
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "mnist_preproc.onnx"
model_quant = "mnist_quantized.onnx"

quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
```

Run the script:
```bash
python3 quantize_mnist.py
```

## Compile
```bash
vai_c_xir -x mnist_quantized.onnx -a arch.json -o compiled_mnist
``` -->
## Test
Create a python script `run_inference_mnist.py` to test the compiled model:
```bash
xxxxxx
```

# KV260

## Boot and Config IP

Boot and set up IP address:
```bash
sudo screen /dev/ttyUSB1 115200
login: petalinux
pass: petalinux
sudo ifconfig eth1 192.168.0.10
```

## Install VOE
```bash
scp vitis_ai_2023.1-r3.5.0.tar.gz voe-0.1.0-py3-none-any.whl onnxruntime_vitisai-1.16.0-py3-none-any.whl petalinux@192.168.0.10:./
sudo tar -xzvf vitis_ai_2023.1-r3.5.0.tar.gz -C /
pip3 install voe*.whl
pip3 install onnxruntime_vitisai*.whl
```

## Startup DPU

```bash
cd dpu/
sudo cp vart.conf /etc/
```

Load DPU:
```bash
sudo xmutil listapps
sudo xmutil unloadappsudo 
sudo xmutil loadapp dpu
```

Check if DPU is working with YOLOX:
```bash
./test_jpeg_yolovx tsd_yolox_pt_acc/tsd_yolox_pt_acc.xmodel stop_sign.jpg
```

Finally, just run the script to run inference:
```bash
python3 run_inference_mnist.py
```
