import numpy as np
import vai_q_onnx
from torch.utils.data import DataLoader
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Callable


class RLCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        data_source: List[np.ndarray],
        num_samples: int = 1000,
        input_name: str = "obs",
        preprocess_fn: Optional[Callable] = None,
    ):
        self.input_name = input_name
        self.num_samples = min(num_samples, len(data_source))
        self.preprocess_fn = preprocess_fn
        self.current_index = 0

        self.data = self._prepare_data(data_source)
        self.data_size = len(self.data)
        self.iterator = iter(self.data)

    def _prepare_data(self, data_source: List[np.ndarray]) -> List[np.ndarray]:
        formatted_data = []
        for data in data_source[: self.num_samples]:
            data = np.array(data, dtype=np.float32)
            if data.shape == (48,):
                data = np.expand_dims(data, axis=0)
            elif data.shape != (1, 48):
                raise ValueError(
                    f"Data shape {data.shape} does not match expected [1,48] or [48]."
                )

            if self.preprocess_fn is not None:
                data = self.preprocess_fn(data)

            if data.shape != (1, 48) or data.dtype != np.float32:
                raise ValueError(
                    f"Processed data shape {data.shape} or type {data.dtype} does not match float32[1,48]."
                )

            formatted_data.append(data)
        return formatted_data

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.current_index >= self.data_size:
            return None
        try:
            data = next(self.iterator)
            self.current_index += 1
            return {self.input_name: data}
        except StopIteration:
            return None


replay_buffer = [np.random.rand(48) for _ in range(1000)]


def preprocess_fn(data):
    return 2.0 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1.0


data_reader = RLCalibrationDataReader(
    replay_buffer,
    num_samples=1000,
    preprocess_fn=preprocess_fn,
)


model_input = "./policy_preproc.onnx"
model_output = "./policy_quantized.onnx"
vai_q_onnx.quantize_static(
    model_input=model_input,
    model_output=model_output,
    calibration_data_reader=data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    per_channel=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
)
print("Quantize finish, quantized models saved in ", model_output)
