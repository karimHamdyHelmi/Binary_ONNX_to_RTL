# Binary Classification ONNX to RTL

Convert binary classification ONNX models (fully connected layers) to RTL and `.mem` files.  Uses `detect_quant_type.py` for autodetection of quantization (int4, int8, int16)

## Requirements

- **Python 3.8+**
- **PyTorch** (for rtl_mapper)
- **ONNX** (from `onnx_lib_path`)
- **convert_model_to_RTL** sibling directory with `onnx_lib`

## ONNX to RTL

```bash
# Convert binary classifier ONNX to RTL + .mem (quantization auto-detected)
python binary_onnx_to_rtl.py --onnx-model path/to/model.onnx --out-dir ./my_ip

# With testbench
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --emit-testbench

# Hierarchical RTL (default): supports any number of FC layers
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --rtl-structure hierarchical

# Fixed 3-block binaryclass_NN template (only for 3 or 6 FC layers)
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --binaryclass-format

# Disable binaryclass_nn style (use relu_layer_array instead of relu_layer)
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --no-binaryclass-nn-style

# Flattened RTL structure (single inlined module)
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --rtl-structure flattened

# Per-layer modules (fc_layer_fc1, fc_out_layer_fc2,...) instead of parameterized
python binary_onnx_to_rtl.py --onnx-model model.onnx --out-dir ./output --no-parameterized-layers

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
- **Any number of layers**: Hierarchical format (default) supports 1, 2, 3, 4, 5, 6, 7+ FC layers. Use `--binaryclass-format` only when you need the fixed 3-block `binaryclass_NN.sv` template (valid for 3 or 6 layers).

## Output

- **RTL modules** in `--out-dir`: `quant_pkg.sv`, `fc_in_layer.sv`, `fc_out_layer.sv`, `relu_layer.sv`, `sigmoid_layer.sv`, `fc_proj_in_weights_rom.sv`, `fc_proj_out_weights_rom.sv`, etc.
- **Memory files** in `--out-dir/mem_files/`: `fc_proj_in_weights.mem`, `fc_proj_in_bias.mem`, etc.
- **Reports**: `mapping_report.txt`, `netlist.json`, `rtl_filelist.f`
