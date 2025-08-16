import onnxruntime as ort
import numpy as np
import time
import argparse
from onnxruntime_extensions import get_library_path

parser = argparse.ArgumentParser(description="Run MNIST inference with ONNXRuntime")
parser.add_argument(
    "--provider",
    choices=["cpu", "dpu"],
    default="cpu",
    help="Execution provider: 'cpu' for CPUExecutionProvider, 'dpu' for VitisAIExecutionProvider",
)
parser.add_argument(
    "--model", type=str, default="policy.onnx", help="Path to ONNX model file"
)
parser.add_argument(
    "--input", type=str, default="input.npy", help="Path to input .npy file"
)
args = parser.parse_args()

provider_map = {"cpu": "CPUExecutionProvider", "dpu": "VitisAIExecutionProvider"}
provider = provider_map[args.provider]

print("Available Execution Provider:", ort.get_available_providers())
print("Selected Execution Provider:", provider)

# input_data = np.load(args.input).astype(np.float32)
input_data = np.random.rand(48).astype(np.float32)
input_data = np.expand_dims(input_data, axis=0)  # [48] -> [1, 48]

options = ort.SessionOptions()
options.log_severity_level = 4  # Log severity level. Applies to session load, initialization, etc. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.

session = ort.InferenceSession(
    args.model,
    sess_options=options,
    providers=[provider],
    provider_options=[
        {
            "config_file": "/usr/bin/vaip_config.json",
            "custom_ops_library": get_library_path(),
        }
    ],
)

input_name = session.get_inputs()[0].name

num_runs = 100
session.run(None, {input_name: input_data})
start = time.time()
# predictions = []
for _ in range(num_runs):
    outputs = session.run(None, {input_name: input_data})
    # predictions.append(np.argmax(outputs[0]))
elapsed = time.time() - start

# print(f"Predicted digit ({args.provider.upper()}): {predictions[-1]}")
print(f"Total time for {num_runs} runs ({args.provider.upper()}): {elapsed:.4f} sec")
print(f"Average time per run ({args.provider.upper()}): {elapsed / num_runs:.6f} sec")
