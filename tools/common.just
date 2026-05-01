MKDIR := 'mkdir -p'
RM-AIVALIDATOR := 'rm -rf ' + AIVALIDATOR_TMP_DIR
DOWN := 'curl -fsSL'
DOWN_OUT := '-o'
PY_FIX_IMPORTS := 'for f in ' + OUT_DIR + '/*_pb2_grpc.py; do [ -f "$f" ] && sed -i "s/^import \\(.*_pb2\\)/from . import \\1/" "$f"; done'

clean:
    {{RM-AIVALIDATOR}}

fetch-proto:
    {{MKDIR}} "{{AIVALIDATOR_TMP_DIR}}"
    {{DOWN}} "https://raw.githubusercontent.com/esclient/protos/{{AIVALIDATOR_PROTO_TAG}}/{{AIVALIDATOR_PROTO_NAME}}" {{DOWN_OUT}} "{{AIVALIDATOR_TMP_DIR}}/{{AIVALIDATOR_PROTO_NAME}}"

run:
    ENV=dev PYTHONPATH=src ./tools/load_envs.sh pdm run run-server

gen-stubs-moderation: fetch-proto
    {{MKDIR}} "{{OUT_DIR}}"
    pdm run python -m grpc_tools.protoc \
      --proto_path="{{AIVALIDATOR_TMP_DIR}}" \
      --python_out="{{OUT_DIR}}" \
      --grpc_python_out="{{OUT_DIR}}" \
      --pyi_out="{{OUT_DIR}}" \
      "{{AIVALIDATOR_TMP_DIR}}/{{AIVALIDATOR_PROTO_NAME}}"
    {{PY_FIX_IMPORTS}}

update-moderation: gen-stubs-moderation clean

format:
    pdm run black .
    pdm run isort .
    pdm run ruff check . --fix

lint:
    pdm run black --check .
    pdm run isort . --check --diff
    pdm run flake8 .
    pdm run ruff check .
    pdm run mypy --strict .

test:
    pdm run pytest tests/

prepare: format lint test

prepare-dataset:
    pdm run python data/prepare_dataset.py

run-qat-training:
    pdm run python trainer/train_qat.py

run-exporter:
    pdm run python trainer/onnx_exporter.py

quantize:
    pdm run python trainer/quantizer.py