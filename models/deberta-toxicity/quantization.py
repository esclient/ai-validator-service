from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
import time
from transformers import DebertaV2TokenizerFast
from huggingface_hub import hf_hub_download
import json, os

model_id = "esclient/deberta-toxicity-model"

config_path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

if isinstance(config.get("extra_special_tokens"), list):
    config["extra_special_tokens"] = {}

os.makedirs("./patched-tokenizer", exist_ok=True)
with open("./patched-tokenizer/tokenizer_config.json", "w") as f:
    json.dump(config, f)

tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
os.system(f"cp {tokenizer_path} ./patched-tokenizer/tokenizer.json")

model_id = "esclient/deberta-toxicity-model"
onnx_path = "./deberta-onnx"
int8_path = "./deberta-int8"

# Step 1 — export to ONNX
tokenizer = DebertaV2TokenizerFast.from_pretrained("./patched-tokenizer", fix_mistral_regex=True)
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True, dtype="float32")
model.save_pretrained(onnx_path, fix_mistral_regex=True)
tokenizer.save_pretrained(onnx_path, fix_mistral_regex=True)

# Step 2 — quantize
quantizer = ORTQuantizer.from_pretrained(onnx_path)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=int8_path, quantization_config=qconfig)

# Step 3 — benchmark the INT8 model 
int8_model = ORTModelForSequenceClassification.from_pretrained(int8_path, file_name="model_quantized.onnx")
int8_tokenizer = DebertaV2TokenizerFast.from_pretrained(int8_path, fix_mistral_regex=True)

inputs = int8_tokenizer("Ihateyou", return_tensors="pt", max_length=64, truncation=True)

# Warm up first (ONNX Runtime JIT-compiles on first run)
int8_model(**inputs)

start = time.perf_counter()
outputs = int8_model(**inputs)
print(f"Latency: {(time.perf_counter() - start)*1000:.1f}ms")