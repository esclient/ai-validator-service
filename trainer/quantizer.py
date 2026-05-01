from pathlib import Path

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from logger.custom_logger import get_logger

ONNX_DIR = Path("models/deberta-qat-onnx-v2")
INT8_DIR = Path("models/deberta-qat-int8-v2")
log = get_logger(__name__)

quantizer = ORTQuantizer.from_pretrained(str(ONNX_DIR))

# avx2 = Intel/AMD CPUs with AVX2 support; use avx512 for newer Intel, arm64 for Apple Silicon / ARM servers
# Be careful when selecting a format: the wrong target can cause
# significant errors in the quantized model's outputs.

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir=str(INT8_DIR), quantization_config=qconfig)
log.info("Quantization complete")
