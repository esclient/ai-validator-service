from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path

ONNX_DIR = Path("models/deberta-qat-onnx-v2")
INT8_DIR  = Path("models/deberta-qat-int8-v2")

quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))

# avx2 = Intel/AMD CPUs with AVX2 support; use avx512 for newer Intel, arm64 for Apple Silicon / ARM servers
# Be careful of what you are using, because the wrong format can cause significant errors in the quantized model's outputs. 

qconfig   = AutoQuantizationConfig.avx2(is_static=False, per_channel=False) 
quantizer.quantize(save_dir=str(INT8_DIR), quantization_config=qconfig)
print("Done")