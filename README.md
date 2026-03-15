# Binary Classification ONNX to RTL

Convert binary classification ONNX models (fully connected layers) to RTL and `.mem` files.  Uses `detect_quant_type.py` for autodetection of quantization (int4, int8, int16)

## Requirements

- **Python 3.8+**
- **PyTorch** (for rtl_mapper)
- **ONNX** (from `onnx_lib_path`)
- **convert_model_to_RTL** sibling directory with `onnx_lib`

## ONNX to RTL

Output format is **binaryclass** (binaryclass_NN module, AXI4-Stream, fc_in/fc_out blocks). Supports **any number of FC layers**; layers are paired consecutively into ceil(N/2) blocks.

```bash
# Convert binary classifier ONNX to RTL + .mem (quantization auto-detected)
python binary_onnx_to_rtl.py --onnx-model path/to/model.onnx --out-dir ./my_ip

# Override quantization
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --weight-format int8

# Inspect ONNX graph structure
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --inspect

# Print raw extracted data only (no RTL generation, --out-dir not required)
python binary_onnx_to_rtl.py --onnx-model model.onnx --dump-extracted
```

## Quantization Detection

`binary_onnx_to_rtl.py` uses `detect_quant_type.py` in the same folder:

```bash
# From ONNX model
python detect_quant_type.py --onnx-model path/to/model.onnx

# From .mem files
python detect_quant_type.py --mem-dir path/to/mem/files

# From checkpoint
python detect_quant_type.py --checkpoint path/to/model.pth
```

## Supported Models

- **FC only**: Gemm, MatMul, QLinearMatMul, QLinearGemm, FusedMatMul, MatMulInteger
- **CNN to FC**: MatMul after Reshape

## Output

- **RTL modules** in `--out-dir`: `quant_pkg.sv`, `fc_in_layer.sv`, `fc_out_layer.sv`, `relu_layer.sv`, `sigmoid_layer.sv`, `binaryclass_NN.sv`, `binaryclass_NN_wrapper.sv`, `fc_proj_in_weights_rom.sv`, `fc_proj_out_weights_rom.sv`, etc. (one proj ROM set per block)
- **Memory files** in `--out-dir`: `fc_proj_in_weights.mem`, `fc_proj_in_bias.mem`, `fc_proj_out_weights.mem`, `fc_proj_out_bias.mem`, etc.
- **Reports**: `mapping_report.txt`, `netlist.json`, `rtl_filelist.f`
