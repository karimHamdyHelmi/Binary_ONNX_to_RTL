# Binary Classification ONNX to RTL

Convert binary classification ONNX models (fully connected layers) to RTL and `.mem` files. Supports 1 output (sigmoid) or 2 outputs (softmax)Uses `detect_quant_type.py` for autodetection of quantization (int4, int8, int16)

## Requirements

- **Python 3.8+**
- **PyTorch**  for rtl_mapper
- **ONNX**  from `onnx_lib_path`
- **convert_model_to_RTL**  sibling directory with `rtl_mapper`

## ONNX to RTL

```bash
# Convert binary classifier ONNX to RTL + .mem (quantization autodetected)
python binary_onnx_to_RTL.py --onnx-model path/to/model.onnx --out-dir ./my_ip

# With testbench
python binary_onnx_to_RTL.py --onnx-model model.onnx --out-dir ./output --emit-testbench

# Override quantization
python binary_onnx_to_RTL.py --onnx-model model.onnx --out-dir ./output --weight-format int8
```

## Quantization Detection

`binary_onnx_to_RTL.py` uses `detect_quant_type.py` in the same folder:

```bash
# From ONNX model
python detect_quant_type.py --onnx-model path/to/model.onnx

# From .mem files
python detect_quant_type.py --mem-dir path/to/mem/files

# From checkpoint
python detect_quant_type.py --checkpoint path/to/model.pth
```

## Supported Models

- **FC only**: Gemm, MatMul, QLinearMatMul, QLinearGemm
- **CNN to FC**: MatMul after Reshape
- **Output**: 1 (sigmoid) or 2 (softmax) classes

## Output

- `src/rtl/systemverilog/` RTL modules, quant_pkg, ROMs
- `src/rtl/systemverilog/mem/` weight and bias `.mem` files
- `mapping_report.txt`, `netlist.json`
