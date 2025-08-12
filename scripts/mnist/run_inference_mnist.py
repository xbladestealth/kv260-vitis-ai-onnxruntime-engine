import onnxruntime as ort
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Run MNIST inference with ONNXRuntime")
parser.add_argument(
    "--provider",
    choices=["cpu", "dpu"],
    default="cpu",
    help="Execution provider: 'cpu' for CPUExecutionProvider, 'dpu' for VitisAIExecutionProvider",
)
parser.add_argument(
    "--model", type=str, default="mnist_cnn.onnx", help="Path to ONNX model file"
)
parser.add_argument(
    "--input", type=str, default="input.npy", help="Path to input .npy file"
)
args = parser.parse_args()

provider_map = {"cpu": "CPUExecutionProvider", "dpu": "VitisAIExecutionProvider"}
provider = provider_map[args.provider]

print(ort.get_available_providers())

input_data = np.load(args.input).astype(np.float32)

session = ort.InferenceSession(
    args.model,
    providers=[provider],
    provider_options=[{"config_file": "/usr/bin/vaip_config.json"}],
)

input_name = session.get_inputs()[0].name
start = time.time()
outputs = session.run(None, {input_name: input_data})
elapsed = time.time() - start
prediction = np.argmax(outputs[0])
print(f"Predicted digit ({args.provider.upper()}): {prediction}")
print(f"Predicted time ({args.provider.upper()}): {elapsed}")
