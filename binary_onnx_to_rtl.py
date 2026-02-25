#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Resolve paths for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_CONVERT_DIR = _SCRIPT_DIR.parent / "convert_model_to_RTL"
_ONNX_LIB = _CONVERT_DIR / "onnx_lib"
if _ONNX_LIB.is_dir() and str(_ONNX_LIB) not in sys.path:
    sys.path.insert(0, str(_ONNX_LIB))
if str(_CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVERT_DIR))


def _load_onnx(onnx_path: Path) -> Any:
    import onnx
    return onnx.load(str(onnx_path))


def _get_initializers_dict(model: Any) -> Dict[str, np.ndarray]:
    from onnx.numpy_helper import to_array
    result = {}
    for init in model.graph.initializer:
        try:
            arr = to_array(init)
            result[init.name] = arr
        except Exception as e:
            LOGGER.warning(f"Could not load initializer {init.name}: {e}")
    return result


def _get_attr(node: Any, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 2:  # INT
                return attr.i
            if attr.type == 5:  # FLOAT
                return attr.f
            if attr.type == 1:  # FLOAT (legacy)
                return attr.f
    return default


def _try_get_matmul_bias_from_add(
    model: Any,
    matmul_node: Any,
    out_features: int,
    name_to_init: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    matmul_out = matmul_node.output[0]
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) < 2:
            continue
        inputs = list(node.input)
        if matmul_out not in inputs:
            continue
        other = inputs[1] if inputs[0] == matmul_out else inputs[0]
        if other in name_to_init:
            b = name_to_init[other].flatten()
            if len(b) == out_features:
                return b.astype(np.float32)
        break
    return None


def _build_value_to_array(model: Any, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    from onnx.numpy_helper import to_array
    result: Dict[str, np.ndarray] = dict(inits)
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2 and len(node.output) >= 1:
            data_name, shape_name = node.input[0], node.input[1]
            if data_name in result and shape_name in result:
                data = result[data_name]
                shape = result[shape_name].flatten().astype(np.int64)
                out = np.reshape(data, shape)
                result[node.output[0]] = out
        elif node.op_type == "Constant" and len(node.output) >= 1:
            for attr in node.attribute:
                if attr.name == "value":
                    result[node.output[0]] = to_array(attr.t)
                    break
    return result


def extract_layers_from_onnx(onnx_path: Path) -> Tuple[List[Any], int]:

    model = _load_onnx(onnx_path)
    inits = _get_initializers_dict(model)
    name_to_init = _build_value_to_array(model, inits)

    layers: List[Dict[str, Any]] = []
    input_size: Optional[int] = None
    fc_counter = 0

    for node in model.graph.node:
        is_qlinear_matmul = node.op_type == "QLinearMatMul"
        is_qlinear_gemm = node.op_type == "QLinearGemm"

        if node.op_type == "Gemm" or is_qlinear_matmul or is_qlinear_gemm:
            fc_counter += 1
            name = f"fc{fc_counter}"
            inputs = list(node.input)

            if is_qlinear_matmul or is_qlinear_gemm:
                if len(inputs) < 8:
                    LOGGER.warning(f"{node.op_type} node {node.name} has < 8 inputs, skipping")
                    continue
                b_name = inputs[3]
                b_scale_name = inputs[4]
                b_zp_name = inputs[5]
                bias_name = inputs[8] if len(inputs) >= 9 else None
            else:
                if len(inputs) < 2:
                    LOGGER.warning(f"Gemm node {node.name} has < 2 inputs, skipping")
                    continue
                b_name = inputs[1]
                bias_name = inputs[2] if len(inputs) > 2 else None

            if b_name not in name_to_init:
                LOGGER.warning(f"Weight {b_name} not in initializers, skipping")
                continue

            w_np = name_to_init[b_name].copy()
            if w_np.ndim != 2:
                LOGGER.warning(f"Weight {b_name} is not 2D, skipping")
                continue

            if is_qlinear_matmul or is_qlinear_gemm:
                b_scale = float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else 1.0
                b_zp = int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else 0
                w_np = (w_np.astype(np.float32) - b_zp) * b_scale
            elif w_np.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                w_np = w_np.astype(np.float32) / 256.0
            else:
                w_np = w_np.astype(np.float32)

            in_features, out_features = w_np.shape
            weight = w_np.T.astype(np.float32)

            if not is_qlinear_matmul and not is_qlinear_gemm:
                trans_b = _get_attr(node, "transB", 0)
                alpha = _get_attr(node, "alpha", 1.0)
                if trans_b:
                    out_features, in_features = w_np.shape[0], w_np.shape[1]
                    weight = w_np.astype(np.float32)
                if alpha != 1.0:
                    weight = weight * float(alpha)

            bias_np: Optional[np.ndarray] = None
            if bias_name and bias_name in name_to_init:
                b_init = name_to_init[bias_name].flatten()
                if is_qlinear_matmul or is_qlinear_gemm:
                    bias_np = b_init.astype(np.float32)
                elif b_init.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    bias_np = b_init.astype(np.float32) / 256.0
                else:
                    bias_np = b_init.astype(np.float32)
                if not is_qlinear_matmul and not is_qlinear_gemm:
                    beta = _get_attr(node, "beta", 1.0)
                    if beta != 1.0:
                        bias_np = bias_np * float(beta)
                if len(bias_np) != out_features:
                    LOGGER.warning(f"Bias shape {bias_np.shape} != out_features {out_features}")
                    bias_np = None

            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=np.float32)

            if input_size is None:
                input_size = in_features

            layers.append({
                "name": name,
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
            })

        elif node.op_type == "MatMul":
            inputs = list(node.input)
            if len(inputs) < 2:
                continue
            b_name = inputs[1]
            if b_name not in name_to_init:
                continue
            w_np = name_to_init[b_name].copy()
            if w_np.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                w_np = w_np.astype(np.float32) / 256.0
            else:
                w_np = w_np.astype(np.float32)
            if w_np.ndim != 2:
                continue
            fc_counter += 1
            in_features, out_features = w_np.shape
            weight = w_np.T.astype(np.float32)
            bias_np = _try_get_matmul_bias_from_add(model, node, out_features, name_to_init)
            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=np.float32)
            if input_size is None:
                input_size = in_features
            layers.append({
                "name": f"fc{fc_counter}",
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
            })

    if input_size is None and layers:
        input_size = layers[0]["in_features"]
    if input_size is None:
        input_size = 256  # Generic fallback for binary classifiers

    return layers, input_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert binary classification ONNX model (Gemm/MatMul FC layers) to RTL and .mem files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        required=True,
        help="Path to ONNX model file (.onnx)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for RTL and .mem files (e.g., ./my_ip)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=256,
        help="Scale factor for quantizing float weights (default: 256)",
    )
    parser.add_argument(
        "--weight-format",
        type=str,
        choices=["int4", "int8", "int16"],
        default=None,
        help="Override auto-detected quantization (default: auto-detect from ONNX via detect_quant_type)",
    )
    parser.add_argument(
        "--data-width",
        type=int,
        default=16,
        help="Data width in bits (default: 16)",
    )
    parser.add_argument(
        "--emit-testbench",
        action="store_true",
        help="Generate testbench",
    )
    parser.add_argument(
        "--emit-rtl-legacy",
        action="store_true",
        help="Also emit legacy rtl/ flow outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    onnx_path = args.onnx_model.resolve()
    out_dir = args.out_dir.resolve()

    if not onnx_path.exists():
        LOGGER.error(f"ONNX model not found: {onnx_path}")
        return 1

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 1. Auto-detect quantization via detect_quant_type 
    from detect_quant_type import detect_quantization_type, quant_type_to_bits

    if args.weight_format:
        quant_type = args.weight_format
        LOGGER.info(f"Using user-specified quantization: {quant_type}")
    else:
        quant_type = detect_quantization_type(onnx_path=onnx_path)
        LOGGER.info(f"Auto-detected quantization: {quant_type}")

    weight_format_bits = quant_type_to_bits(quant_type)

    # 2. Extract FC layers from ONNX
    LOGGER.info("Extracting layers from ONNX...")
    layers_raw, input_size = extract_layers_from_onnx(onnx_path)

    if not layers_raw:
        LOGGER.error(
            "No Gemm/MatMul/QLinearMatMul/QLinearGemm layers found in ONNX model. "
            "Binary classifier expects FC layers."
        )
        return 1

    LOGGER.info(f"Found {len(layers_raw)} linear layers, input_size={input_size}")
    last_out = layers_raw[-1]["out_features"]
    LOGGER.info(f"Output classes: {last_out} (binary: 1 or 2)")

    # 3. Build LayerInfo and quantize
    from rtl_mapper import (
        LayerInfo,
        float_to_int,
        generate_quant_pkg_style_weight_mem,
        generate_quant_pkg_style_bias_mem,
        write_embedded_rtl_templates,
        generate_weight_rom,
        generate_bias_rom,
        generate_fc_layer_wrapper,
        generate_fc_out_layer,
        generate_top_module,
        generate_testbench,
        generate_mapping_report,
        generate_netlist_json,
        emit_legacy_rtl_outputs,
    )

    layers: List[LayerInfo] = []
    for lr in layers_raw:
        w_np = lr["weight"].astype(np.float32)
        b_np = lr["bias"].astype(np.float32)
        layer = LayerInfo(
            name=lr["name"],
            layer_type="linear",
            in_features=lr["in_features"],
            out_features=lr["out_features"],
            weight=torch.from_numpy(w_np),
            bias=torch.from_numpy(b_np),
        )
        layers.append(layer)

    flatten = LayerInfo(name="flatten_1", layer_type="flatten", out_shape=(1, input_size))
    full_layers: List[LayerInfo] = [flatten]
    for i, layer in enumerate(layers):
        full_layers.append(layer)
        if i < len(layers) - 1:
            full_layers.append(LayerInfo(name=f"relu_{i+1}", layer_type="relu"))
    layers = full_layers

    # 4. Setup output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    sv_dir = out_dir / "src" / "rtl" / "systemverilog"
    sv_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = sv_dir / "mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    tb_sim_dir = out_dir / "tb" / "sim"
    tb_sim_dir.mkdir(parents=True, exist_ok=True)

    # 5. Generate .mem files
    LOGGER.info(f"Writing .mem files (int{weight_format_bits})...")
    for layer in layers:
        if layer.layer_type != "linear":
            continue
        w_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        b_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((layer.out_features or 0,), dtype=np.float32)
        wq = float_to_int(w_np, args.scale, weight_format_bits)
        bq = float_to_int(b_np, args.scale, weight_format_bits)
        weight_mem_path = mem_dir / f"{layer.name}_weights_packed.mem"
        bias_mem_path = mem_dir / f"{layer.name}_biases.mem"
        generate_quant_pkg_style_weight_mem(
            wq, weight_mem_path, layer.name,
            layer.in_features or 0, layer.out_features or 0,
            weight_format_bits,
        )
        generate_quant_pkg_style_bias_mem(bq, bias_mem_path, layer.out_features or 0, weight_format_bits)
        LOGGER.info(f"  {layer.name}: {layer.in_features} -> {layer.out_features}")

    # 6. Emit legacy RTL if requested
    if args.emit_rtl_legacy:
        legacy_dir = out_dir.parent / "legacy_out" if out_dir.name == "my_ip" else out_dir / "legacy"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        emit_legacy_rtl_outputs(
            legacy_rtl_dir=legacy_dir,
            layers=layers,
            scale=args.scale,
            bits_list=(4, 8, 16),
            write_sv=False,
        )

    # 7. Write embedded RTL templates
    LOGGER.info("Writing RTL templates...")
    write_embedded_rtl_templates(sv_dir, weight_format_bits)

    # 8. Generate ROM and layer modules
    LOGGER.info("Generating ROM and layer modules...")
    for layer in layers:
        if layer.layer_type == "linear":
            generate_weight_rom(layer.name, layer.in_features or 0, layer.out_features or 0, weight_format_bits, sv_dir)
            generate_bias_rom(layer.name, layer.out_features or 0, weight_format_bits, sv_dir)
            if layer.name == "fc1":
                generate_fc_layer_wrapper(
                    layer.name, layer.in_features or 0, layer.out_features or 0,
                    args.data_width, weight_format_bits, sv_dir,
                )
            else:
                generate_fc_out_layer(
                    layer.name, layer.out_features or 0, layer.in_features or 0, sv_dir,
                )

    # 9. Generate top module
    model_name = onnx_path.stem
    LOGGER.info("Generating top module...")
    generate_top_module(model_name, layers, input_size, args.data_width, weight_format_bits, sv_dir)

    # 10. Testbench
    if args.emit_testbench:
        last_fc = next((l for l in reversed(layers) if l.layer_type == "linear"), None)
        if last_fc:
            generate_testbench(model_name, input_size, last_fc.out_features or 0, weight_format_bits, tb_sim_dir)

    # 11. Reports
    frac_bits = 8
    generate_mapping_report(out_dir, model_name, layers, args.scale, args.data_width, weight_format_bits, 32, frac_bits)
    generate_netlist_json(out_dir, model_name, layers)

    LOGGER.info(f"Binary classifier RTL generation complete! Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
