#!/usr/bin/env python3
"""
Binary ONNX to RTL Converter
============================
Converts binary classification ONNX models (Gemm/MatMul FC layers) to RTL
and .mem files for hardware synthesis. Supports Gemm, MatMul, QLinearMatMul,
QLinearGemm, FusedMatMul, and MatMulInteger op types.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------
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
# rtl_mapper is in same directory as this script (Binary_ONNX_to_RTL)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# -----------------------------------------------------------------------------
# ONNX Loading and Graph Utilities
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Bias Extraction Helpers (MatMul / MatMulInteger chains)
# -----------------------------------------------------------------------------

def _try_get_matmul_bias_from_add(
    model: Any,
    matmul_node: Any,
    out_features: int,
    name_to_init: Dict[str, np.ndarray],
    raw: bool = False,
) -> Optional[Tuple[np.ndarray, Optional[str]]]:
    """Returns (bias_array, init_name) or None. init_name is the ONNX initializer name."""
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
                arr = b.copy() if raw else b.astype(np.float32)
                return (arr, other)
        break
    return None


def _try_get_bias_from_add_chain(
    model: Any,
    start_output: str,
    out_features: int,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
    raw: bool = False,
) -> Optional[Tuple[np.ndarray, Optional[str]]]:
    """Follow output through Mul/Cast to find Add with constant bias. Returns (bias_array, init_name) or None."""
    if visited is None:
        visited = set()
    if start_output in visited:
        return None
    visited.add(start_output)
    for node in model.graph.node:
        if start_output not in list(node.input):
            continue
        if node.op_type == "Add" and len(node.input) >= 2:
            inputs = list(node.input)
            other = inputs[1] if inputs[0] == start_output else inputs[0]
            if other in name_to_init:
                arr = name_to_init[other]
                b = arr.flatten()
                if len(b) == out_features:
                    if raw:
                        return (b.copy(), other)
                    if arr.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                        return ((b.astype(np.float32) - 0) / 128.0, other)
                    if arr.dtype == np.int32:
                        return (b.astype(np.float32) / 16384.0, other)
                    return (b.astype(np.float32), other)
        if node.op_type in ("Mul", "Cast") and node.output:
            found = _try_get_bias_from_add_chain(
                model, node.output[0], out_features, name_to_init, visited, raw
            )
            if found is not None:
                return found
    return None


def _try_get_scale_from_mul_chain(
    model: Any,
    start_output: str,
    name_to_init: Dict[str, np.ndarray],
    visited: Optional[set] = None,
) -> Optional[Tuple[float, Optional[str]]]:
    """Follow output through Mul to find scale constant (for MatMulInteger dequant). Returns (scale, init_name) or None."""
    if visited is None:
        visited = set()
    if start_output in visited:
        return None
    visited.add(start_output)
    for node in model.graph.node:
        if start_output not in list(node.input):
            continue
        if node.op_type == "Mul" and len(node.input) >= 2:
            inputs = list(node.input)
            other = inputs[1] if inputs[0] == start_output else inputs[0]
            if other in name_to_init:
                scale_arr = name_to_init[other]
                if scale_arr.size == 1:
                    return (float(scale_arr.flatten()[0]), other)
        if node.op_type in ("Mul", "Cast") and node.output:
            found = _try_get_scale_from_mul_chain(model, node.output[0], name_to_init, visited)
            if found is not None:
                return found
    return None


# -----------------------------------------------------------------------------
# Value/Shape Resolution (Reshape, Constant)
# -----------------------------------------------------------------------------

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


def _get_node_op_type(node: Any) -> str:
    """Return op_type, including domain prefix for non-standard ops."""
    domain = getattr(node, "domain", "") or ""
    if domain and domain != "ai.onnx":
        return f"{domain}::{node.op_type}"
    return node.op_type


# -----------------------------------------------------------------------------
# Final Activation Detection (trace from graph output backwards)
# -----------------------------------------------------------------------------

# Activation ops we support for final layer
_SUPPORTED_FINAL_ACTIVATIONS = frozenset({"Sigmoid", "Relu", "Softmax", "Tanh"})
# Ops that just pass through (trace to input)
_PASSTHROUGH_OPS = frozenset({"Identity", "Cast", "Reshape", "Squeeze", "Unsqueeze"})


# Activations we support in RTL (per-layer)
_SUPPORTED_ACTIVATIONS = frozenset({"Relu", "Sigmoid", "Tanh", "HardSigmoid"})
# Ops to follow when tracing forward (before activation)
_FORWARD_PASSTHROUGH = frozenset({"Add", "Mul", "Cast", "Identity", "Reshape", "Squeeze", "Unsqueeze", "DynamicQuantizeLinear"})


def _trace_activation_forward(
    model: Any,
    start_name: str,
    input_to_nodes: Dict[str, List[Any]],
    visited: Optional[set] = None,
) -> Optional[str]:
    """Trace forward from start_name to find the first activation op. Returns op_type or None."""
    if visited is None:
        visited = set()
    if start_name in visited:
        return None
    visited.add(start_name)
    for consumer in input_to_nodes.get(start_name, []):
        op = consumer.op_type
        if op in _SUPPORTED_ACTIVATIONS:
            return op
        if op in _FORWARD_PASSTHROUGH and consumer.output:
            found = _trace_activation_forward(model, consumer.output[0], input_to_nodes, visited)
            if found is not None:
                return found
        if op in ("Gemm", "MatMul", "QLinearMatMul", "QLinearGemm", "MatMulInteger"):
            break  # Next FC, stop tracing
    return None


def extract_per_layer_activations(model: Any, fc_output_names: List[str]) -> List[Optional[str]]:
    """For each FC output, trace forward to find the activation that follows. Returns list of activation op_types."""
    input_to_nodes: Dict[str, List[Any]] = {}
    for node in model.graph.node:
        for inp in node.input:
            input_to_nodes.setdefault(inp, []).append(node)
    result: List[Optional[str]] = []
    for out_name in fc_output_names:
        act = _trace_activation_forward(model, out_name, input_to_nodes)
        result.append(act)
    return result


def _detect_final_activation(model: Any) -> Optional[str]:
    """Trace from graph output backwards to find the last activation op (Sigmoid, Relu, etc.).
    Returns the op_type (e.g. 'Sigmoid') or None if not found."""
    if not model.graph.output:
        return None
    output_name = model.graph.output[0].name

    # Build output_name -> node mapping
    output_to_node: Dict[str, Any] = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    current = output_name
    visited: set = set()

    while current and current not in visited:
        visited.add(current)
        node = output_to_node.get(current)
        if node is None:
            return None
        op = node.op_type
        if op in _SUPPORTED_FINAL_ACTIVATIONS:
            return op
        if op in _PASSTHROUGH_OPS and node.input:
            current = node.input[0]
        else:
            return None
    return None


# -----------------------------------------------------------------------------
# Layer Extraction from ONNX Graph
# -----------------------------------------------------------------------------

def extract_layers_from_onnx(onnx_path: Path, raw: bool = False) -> Tuple[List[Any], int, Optional[str]]:
    """Extract FC layers from ONNX. If raw=True, preserve original dtypes/values (no dequantization)."""
    model = _load_onnx(onnx_path)
    inits = _get_initializers_dict(model)
    name_to_init = _build_value_to_array(model, inits)

    layers: List[Dict[str, Any]] = []
    input_size: Optional[int] = None
    fc_counter = 0

    for node in model.graph.node:
        op_type = _get_node_op_type(node)
        is_qlinear_matmul = node.op_type == "QLinearMatMul"
        is_qlinear_gemm = node.op_type == "QLinearGemm"
        is_fused_matmul = op_type == "com.microsoft::FusedMatMul"

        # --- Gemm / QLinearMatMul / QLinearGemm / FusedMatMul ---
        if node.op_type == "Gemm" or is_qlinear_matmul or is_qlinear_gemm or is_fused_matmul:
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
            elif is_fused_matmul:
                if len(inputs) < 2:
                    LOGGER.warning(f"FusedMatMul node {node.name} has < 2 inputs, skipping")
                    continue
                b_name = inputs[1]  # B = weight matrix
                bias_name = inputs[2] if len(inputs) >= 3 else None
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

            if raw:
                weight = w_np
                in_features, out_features = w_np.shape
                if not is_qlinear_matmul and not is_qlinear_gemm and not is_fused_matmul:
                    trans_b = _get_attr(node, "transB", 0)
                    if trans_b:
                        out_features, in_features = w_np.shape[0], w_np.shape[1]
            else:
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

                if not is_qlinear_matmul and not is_qlinear_gemm and not is_fused_matmul:
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
                if len(b_init) != out_features:
                    LOGGER.warning(f"Bias shape {b_init.shape} != out_features {out_features}")
                elif raw:
                    bias_np = b_init.copy()
                elif is_qlinear_matmul or is_qlinear_gemm:
                    bias_np = b_init.astype(np.float32)
                elif b_init.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    bias_np = b_init.astype(np.float32) / 256.0
                else:
                    bias_np = b_init.astype(np.float32)
                if not raw and bias_np is not None and not is_qlinear_matmul and not is_qlinear_gemm:
                    beta = _get_attr(node, "beta", 1.0)
                    if beta != 1.0:
                        bias_np = bias_np * float(beta)

            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)

            if input_size is None:
                input_size = in_features

            layer_entry = {
                "name": name,
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
            }
            if raw:
                layer_entry["weight_init_name"] = b_name
                layer_entry["bias_init_name"] = bias_name if (bias_name and bias_name in name_to_init) else None
                if is_qlinear_matmul or is_qlinear_gemm:
                    layer_entry["quant_params"] = {
                        "b_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "b_zero_point": int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else None,
                    }
                else:
                    layer_entry["quant_params"] = None
            else:
                # Non-raw: add quant_params for RTL scale computation
                if is_qlinear_matmul or is_qlinear_gemm:
                    layer_entry["quant_params"] = {
                        "weight_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "b_scale": float(name_to_init[b_scale_name].flatten()[0]) if b_scale_name in name_to_init else None,
                        "b_zero_point": int(name_to_init[b_zp_name].flatten()[0]) if b_zp_name in name_to_init else None,
                    }
                else:
                    layer_entry["quant_params"] = None
            layers.append(layer_entry)

        # --- MatMul (float or pre-quantized) ---
        elif node.op_type == "MatMul":
            inputs = list(node.input)
            if len(inputs) < 2:
                continue
            b_name = inputs[1]
            if b_name not in name_to_init:
                continue
            w_np = name_to_init[b_name].copy()
            if w_np.ndim != 2:
                continue
            fc_counter += 1
            in_features, out_features = w_np.shape
            bias_init_name: Optional[str] = None
            if raw:
                weight = w_np
                bias_result = _try_get_matmul_bias_from_add(model, node, out_features, name_to_init, raw=True)
                bias_np = bias_result[0] if bias_result else None
                if bias_result:
                    bias_init_name = bias_result[1]
            else:
                if w_np.dtype in (np.int8, np.uint8, np.int16, np.uint16):
                    w_np = w_np.astype(np.float32) / 256.0
                else:
                    w_np = w_np.astype(np.float32)
                weight = w_np.T.astype(np.float32)
                bias_result = _try_get_matmul_bias_from_add(model, node, out_features, name_to_init)
                bias_np = bias_result[0] if bias_result else None
            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)
            if input_size is None:
                input_size = in_features
            layer_entry: Dict[str, Any] = {
                "name": f"fc{fc_counter}",
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
            }
            if raw:
                layer_entry["weight_init_name"] = b_name
                layer_entry["bias_init_name"] = bias_init_name
                layer_entry["quant_params"] = None
            else:
                layer_entry["quant_params"] = None  # MatMul float
            layers.append(layer_entry)

        # --- MatMulInteger (dynamic int8 quantization) ---
        elif node.op_type == "MatMulInteger":
            # Dynamic quantization: int8 activations * int8 weights -> int32
            inputs = list(node.input)
            if len(inputs) < 2:
                continue
            b_name = inputs[1]
            if b_name not in name_to_init:
                continue
            w_np = name_to_init[b_name].copy()
            if w_np.ndim != 2:
                continue
            in_features, out_features = w_np.shape
            bias_init_name_mmi: Optional[str] = None
            if raw:
                weight = w_np
                bias_result = _try_get_bias_from_add_chain(model, node.output[0], out_features, name_to_init, raw=True)
                bias_np = bias_result[0] if bias_result else None
                if bias_result:
                    bias_init_name_mmi = bias_result[1]
            else:
                if len(inputs) >= 4 and inputs[3] in name_to_init:
                    zp_arr = name_to_init[inputs[3]]
                    if zp_arr.size == 1:
                        b_zp = int(zp_arr.flatten()[0])
                        w_np = (w_np.astype(np.float32) - b_zp) / 128.0
                    elif zp_arr.size == w_np.shape[1]:  # per-column zero point
                        w_np = (w_np.astype(np.float32) - zp_arr.astype(np.float32).reshape(1, -1)) / 128.0
                    else:
                        w_np = w_np.astype(np.float32) / 128.0
                else:
                    w_np = w_np.astype(np.float32) / 128.0
                weight = w_np.T.astype(np.float32)
                bias_result = _try_get_bias_from_add_chain(model, node.output[0], out_features, name_to_init)
                bias_np = bias_result[0] if bias_result else None
            if bias_np is None:
                bias_np = np.zeros((out_features,), dtype=w_np.dtype if raw else np.float32)
            if input_size is None:
                input_size = in_features
            fc_counter += 1
            layer_entry_mmi: Dict[str, Any] = {
                "name": f"fc{fc_counter}",
                "weight": weight,
                "bias": bias_np,
                "in_features": in_features,
                "out_features": out_features,
                "fc_output_name": node.output[0] if node.output else None,
            }
            if raw:
                layer_entry_mmi["weight_init_name"] = b_name
                layer_entry_mmi["bias_init_name"] = bias_init_name_mmi
                b_zp_val: Any = None
                if len(inputs) >= 4 and inputs[3] in name_to_init:
                    zp_arr = name_to_init[inputs[3]]
                    if zp_arr.size == 1:
                        b_zp_val = int(zp_arr.flatten()[0])
                    else:
                        b_zp_val = zp_arr  # per-column, show as array
                scale_result = _try_get_scale_from_mul_chain(model, node.output[0], name_to_init)
                scale_val = scale_result[0] if scale_result else None
                scale_init_name = scale_result[1] if scale_result else None
                # Also try weight_scale from initializers (e.g. fc_proj_in_MatMul_W_scale)
                weight_scale_init = b_name.replace("_quantized", "_scale") if "_quantized" in b_name else None
                weight_scale_val = float(name_to_init[weight_scale_init].flatten()[0]) if weight_scale_init and weight_scale_init in name_to_init else None
                layer_entry_mmi["quant_params"] = {
                    "b_zero_point": b_zp_val,
                    "b_zero_point_init_name": inputs[3] if len(inputs) >= 4 else None,
                    "scale": scale_val,
                    "scale_init_name": scale_init_name,
                    "weight_scale": weight_scale_val,
                    "weight_scale_init_name": weight_scale_init if weight_scale_init in name_to_init else None,
                }
            else:
                # Non-raw: add quant_params for RTL scale computation (MatMulInteger)
                weight_scale_init = b_name.replace("_quantized", "_scale") if "_quantized" in b_name else None
                weight_scale_val = float(name_to_init[weight_scale_init].flatten()[0]) if weight_scale_init and weight_scale_init in name_to_init else None
                b_zp_val_mmi = int(name_to_init[inputs[3]].flatten()[0]) if len(inputs) >= 4 and inputs[3] in name_to_init and name_to_init[inputs[3]].size == 1 else None
                layer_entry_mmi["quant_params"] = {
                    "weight_scale": weight_scale_val,
                    "b_zero_point": b_zp_val_mmi,
                } if weight_scale_val is not None else None
            layers.append(layer_entry_mmi)

    if input_size is None and layers:
        input_size = layers[0]["in_features"]
    if input_size is None:
        input_size = 256  # Generic fallback for binary classifiers

    final_activation = _detect_final_activation(model)

    # Per-layer activation detection (trace forward from each FC output)
    fc_output_names = [l.get("fc_output_name") for l in layers if l.get("fc_output_name")]
    per_layer_activations = extract_per_layer_activations(model, fc_output_names) if fc_output_names else []
    for i, act in enumerate(per_layer_activations):
        if i < len(layers):
            layers[i]["activation"] = act  # Relu, Sigmoid, Tanh, etc. or None

    return layers, input_size, final_activation


# -----------------------------------------------------------------------------
# CLI Entry Point and Main Pipeline
# -----------------------------------------------------------------------------

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
        default=None,
        help="Output directory for RTL and .mem files (e.g., ./my_ip). Not required with --dump-extracted or --inspect.",
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
        "--rtl-structure",
        type=str,
        choices=["hierarchical", "flattened"],
        default="hierarchical",
        help="RTL structure: 'hierarchical' (separate modules) or 'flattened' (single inlined module). Default: hierarchical",
    )
    parser.add_argument(
        "--parameterized-layers",
        action="store_true",
        default=True,
        help="Use parameterized fc_in_layer/fc_out_layer (default: True).",
    )
    parser.add_argument(
        "--no-parameterized-layers",
        action="store_false",
        dest="parameterized_layers",
        help="Use per-layer fc_layer_fc1, fc_out_layer_fcN instead of parameterized layers.",
    )
    parser.add_argument(
        "--no-binaryclass-nn-style",
        action="store_false",
        dest="binaryclass_nn_style",
        default=True,
        help="Use binaryclass_nn style (fc_in->fc_out->relu, sigmoid at end). Default: True.",
    )
    parser.add_argument(
        "--binaryclass-format",
        action="store_true",
        help="Use fixed 3-block binaryclass_NN template (only for 3 or 6 FC layers). "
        "By default, hierarchical format supports any number of layers.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print ONNX graph structure (node types, inputs) and exit",
    )
    parser.add_argument(
        "--dump-extracted",
        action="store_true",
        help="Only print raw extracted data from ONNX (layers, dimensions, weights, biases) and exit",
    )

    args = parser.parse_args()
    onnx_path = args.onnx_model.resolve()
    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = None

    # Validate input
    if not onnx_path.exists():
        LOGGER.error(f"ONNX model not found: {onnx_path}")
        return 1
    if not args.dump_extracted and not args.inspect and out_dir is None:
        parser.error("--out-dir is required for RTL generation (omit with --dump-extracted or --inspect)")

    # Inspect mode: print graph structure and exit
    if args.inspect:
        model = _load_onnx(onnx_path)
        node_types: Dict[str, int] = {}
        for node in model.graph.node:
            op = _get_node_op_type(node)
            node_types[op] = node_types.get(op, 0) + 1
        print("Node types:", ", ".join(f"{op}({n})" for op, n in sorted(node_types.items())))
        print("\nNodes:")
        for node in model.graph.node:
            op = _get_node_op_type(node)
            ins = ", ".join(node.input[:4]) + ("..." if len(node.input) > 4 else "")
            print(f"  {op}: {node.name} <- [{ins}]")
        return 0

    # Dump-extracted mode: print raw extracted data (as stored in ONNX, no dequantization) and exit
    if args.dump_extracted:
        model = _load_onnx(onnx_path)
        inits = _get_initializers_dict(model)
        layers_raw, input_size, final_activation = extract_layers_from_onnx(onnx_path, raw=True)
        if not layers_raw:
            LOGGER.error("No FC layers found in ONNX model.")
            return 1

        _FC_OP_TYPES = frozenset({"Gemm", "MatMul", "QLinearMatMul", "QLinearGemm", "com.microsoft::FusedMatMul", "MatMulInteger"})

        print("=== Raw Extracted Data from ONNX (as stored, no dequantization) ===\n")
        print(f"input_size: {input_size}")
        print(f"final_activation: {final_activation}\n")

        print("--- FC Layers ---\n")
        for lr in layers_raw:
            w = lr["weight"]
            b = lr["bias"]
            print(f"Layer {lr['name']}: in={lr['in_features']}, out={lr['out_features']}")
            print(f"  weight: shape {w.shape}, dtype {w.dtype}, min={w.min()}, max={w.max()}")
            if "weight_init_name" in lr:
                print(f"  weight_init_name: {lr['weight_init_name']}")
            print(f"  bias: shape {b.shape}, dtype {b.dtype}, min={b.min()}, max={b.max()}")
            if "bias_init_name" in lr and lr["bias_init_name"]:
                print(f"  bias_init_name: {lr['bias_init_name']}")
            if "quant_params" in lr and lr["quant_params"]:
                qp = lr["quant_params"]
                print(f"  quant_params:")
                for k, v in qp.items():
                    if v is not None:
                        if isinstance(v, np.ndarray):
                            print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
                        else:
                            print(f"    {k}: {v}")
            print()

        print("--- Initializers (all) ---\n")
        for name, arr in sorted(inits.items()):
            print(f"  {name}: shape {arr.shape}, dtype {arr.dtype}")

        print("\n--- Non-FC Ops ---\n")
        for node in model.graph.node:
            op = _get_node_op_type(node)
            if op not in _FC_OP_TYPES:
                ins = ", ".join(node.input[:4]) + ("..." if len(node.input) > 4 else "")
                outs = ", ".join(node.output[:2]) + ("..." if len(node.output) > 2 else "")
                print(f"  {op}: {node.name}")
                print(f"    inputs: [{ins}]")
                print(f"    outputs: [{outs}]")
        return 0

    # Step 1: Auto-detect quantization via detect_quant_type 
    from detect_quant_type import detect_quantization_type, quant_type_to_bits

    if args.weight_format:
        quant_type = args.weight_format
        LOGGER.info(f"Using user-specified quantization: {quant_type}")
    else:
        quant_type = detect_quantization_type(onnx_path=onnx_path)
        LOGGER.info(f"Auto-detected quantization: {quant_type}")

    weight_format_bits = quant_type_to_bits(quant_type)

    # Step 2: Extract FC layers from ONNX
    LOGGER.info("Extracting layers from ONNX...")
    layers_raw, input_size, final_activation = extract_layers_from_onnx(onnx_path)

    if not layers_raw:
        # Diagnostic: list node types in the model
        model = _load_onnx(onnx_path)
        node_types: Dict[str, int] = {}
        for node in model.graph.node:
            op = _get_node_op_type(node)
            node_types[op] = node_types.get(op, 0) + 1
        summary = ", ".join(f"{op}({n})" for op, n in sorted(node_types.items()))
        LOGGER.error(
            "No Gemm/MatMul/QLinearMatMul/QLinearGemm/FusedMatMul/MatMulInteger layers found in ONNX model. "
            "Binary classifier expects FC layers."
        )
        LOGGER.info(f"Node types in model: {summary}")
        return 1

    LOGGER.info(f"Found {len(layers_raw)} linear layers, input_size={input_size}")
    last_out = layers_raw[-1]["out_features"]
    LOGGER.info(f"Output classes: {last_out} (binary: 1 or 2)")
    if final_activation:
        LOGGER.info(f"Detected final activation from ONNX graph: {final_activation}")
    else:
        LOGGER.info("No final activation detected in ONNX graph; will use default for binary classifier (Sigmoid)")

    # Step 3: Build LayerInfo objects and add flatten/ReLU structure
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
        generate_proj_roms,
        generate_proj_mem_files,
        generate_fc_in_layer_parameterized,
        generate_fc_out_layer_parameterized,
        generate_top_module,
        generate_axi4_stream_wrapper,
        generate_flattened_top_module,
        generate_wrapper_module,
        generate_testbench,
        generate_mapping_report,
        generate_netlist_json,
        generate_rtl_filelist,
        emit_legacy_rtl_outputs,
        emit_binaryclass_nn_format,
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
            quant_params=lr.get("quant_params"),
            activation=lr.get("activation"),
        )
        layers.append(layer)

    flatten = LayerInfo(name="flatten_1", layer_type="flatten", out_shape=(1, input_size))
    full_layers: List[LayerInfo] = [flatten]
    for i, layer in enumerate(layers):
        full_layers.append(layer)
        if i < len(layers) - 1:
            full_layers.append(LayerInfo(name=f"relu_{i+1}", layer_type="relu"))
    layers = full_layers

    # Step 4: Setup output directories (binaryclass_nn format: flat layout)
    out_dir.mkdir(parents=True, exist_ok=True)
    sv_dir = out_dir
    sv_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = out_dir / "mem_files"
    mem_dir.mkdir(parents=True, exist_ok=True)
    tb_sim_dir = out_dir / "tb" / "sim"
    tb_sim_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Generate .mem files (binaryclass_nn format: mem_files/fc1_weights.mem)
    LOGGER.info(f"Writing .mem files (int{weight_format_bits})...")
    for layer in layers:
        if layer.layer_type != "linear":
            continue
        w_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        b_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((layer.out_features or 0,), dtype=np.float32)
        wq = float_to_int(w_np, args.scale, weight_format_bits)
        bq = float_to_int(b_np, args.scale, weight_format_bits)
        weight_mem_path = mem_dir / f"{layer.name}_weights.mem"
        bias_mem_path = mem_dir / f"{layer.name}_biases.mem"
        generate_quant_pkg_style_weight_mem(
            wq, weight_mem_path, layer.name,
            layer.in_features or 0, layer.out_features or 0,
            weight_format_bits,
        )
        generate_quant_pkg_style_bias_mem(bq, bias_mem_path, layer.out_features or 0, weight_format_bits)
        LOGGER.info(f"  {layer.name}: {layer.in_features} -> {layer.out_features}")

    # Step 6: Emit legacy RTL if requested
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

    model_name = onnx_path.stem
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    # binaryclass_nn format: fixed 3-block template, only when --binaryclass-format and 3 or 6 layers
    # By default: hierarchical format supports any number of layers
    use_binaryclass_nn = (
        args.binaryclass_format
        and len(linear_layers) in (3, 6)
        and all(l.name == f"fc{i+1}" for i, l in enumerate(linear_layers))
    )
    use_flattened = args.rtl_structure == "flattened" and not use_binaryclass_nn

    if use_binaryclass_nn:
        LOGGER.info("Generating binaryclass_nn format (binaryclass_NN, AXI4-Stream, reference structure)...")
        emit_binaryclass_nn_format(out_dir, layers, input_size, weight_format_bits, args.scale)
        generate_mapping_report(out_dir, model_name, layers, args.scale, args.data_width, weight_format_bits, 32, 8, final_activation=final_activation)
        generate_netlist_json(out_dir, model_name, layers, final_activation=final_activation)
        generate_rtl_filelist(out_dir, model_name, layers, binaryclass_nn_format=True)
        LOGGER.info(f"Binaryclass_nn format RTL generation complete! Output: {out_dir}")
        return 0

    if use_flattened and len(linear_layers) < 3:
        LOGGER.warning(
            f"Flattened RTL requires at least 3 FC layers; found {len(linear_layers)}. "
            "Falling back to hierarchical structure."
        )
        use_flattened = False

    # Step 7: Write embedded RTL templates (quant_pkg always needed; submodules only for hierarchical)
    LOGGER.info("Writing RTL templates...")
    write_embedded_rtl_templates(sv_dir, weight_format_bits, write_submodules=not use_flattened)

    if use_flattened:
        # 8a. Flattened: single inlined top module + wrapper (no separate ROM/layer modules)
        LOGGER.info("Generating flattened RTL structure...")
        generate_flattened_top_module(
            model_name, layers, input_size, args.data_width, weight_format_bits, sv_dir
        )
        generate_wrapper_module(model_name, layers, weight_format_bits, sv_dir)
    else:
        # 8b. Hierarchical: ROM, layer, and top modules
        LOGGER.info("Generating hierarchical RTL structure...")
        if args.parameterized_layers:
            LOGGER.info("  Using parameterized fc_in_layer / fc_out_layer...")
            # binaryclass_nn layout: .mem in out_dir root, ROM $readmemh uses simple filename
            generate_proj_mem_files(layers, out_dir, args.scale, weight_format_bits)
            generate_proj_roms(layers, weight_format_bits, sv_dir, out_dir, mem_path_prefix="")
            generate_fc_in_layer_parameterized(layers, weight_format_bits, sv_dir)
            generate_fc_out_layer_parameterized(layers, weight_format_bits, sv_dir)
            generate_top_module(
                model_name, layers, input_size, args.data_width, weight_format_bits, sv_dir,
                use_parameterized_layers=True,
                use_binaryclass_nn_style=args.binaryclass_nn_style,
                final_activation=final_activation,
                python_scale=args.scale,
            )
            generate_axi4_stream_wrapper(model_name, layers, weight_format_bits, sv_dir)
        else:
            for layer in layers:
                if layer.layer_type == "linear":
                    generate_weight_rom(layer.name, layer.in_features or 0, layer.out_features or 0, weight_format_bits, sv_dir, mem_subdir="mem_files")
                    generate_bias_rom(layer.name, layer.out_features or 0, weight_format_bits, sv_dir, mem_subdir="mem_files")
                    if layer.name == "fc1":
                        generate_fc_layer_wrapper(
                            layer.name, layer.in_features or 0, layer.out_features or 0,
                            args.data_width, weight_format_bits, sv_dir,
                        )
                    else:
                        generate_fc_out_layer(
                            layer.name, layer.out_features or 0, layer.in_features or 0, sv_dir,
                        )

            LOGGER.info("Generating top module...")
            generate_top_module(
                model_name, layers, input_size, args.data_width, weight_format_bits, sv_dir,
                use_parameterized_layers=False,
                final_activation=final_activation,
                python_scale=args.scale,
            )
            generate_axi4_stream_wrapper(model_name, layers, weight_format_bits, sv_dir)

    # Step 8: Testbench (optional)
    if args.emit_testbench:
        last_fc = next((l for l in reversed(layers) if l.layer_type == "linear"), None)
        if last_fc:
            generate_testbench(model_name, input_size, last_fc.out_features or 0, weight_format_bits, tb_sim_dir)

    # Step 9: Reports (mapping report, netlist JSON, RTL filelist)
    frac_bits = 8
    generate_mapping_report(out_dir, model_name, layers, args.scale, args.data_width, weight_format_bits, 32, frac_bits, final_activation=final_activation)
    generate_netlist_json(out_dir, model_name, layers, final_activation=final_activation)
    generate_rtl_filelist(
        out_dir, model_name, layers,
        parameterized_layers=(not use_flattened and args.parameterized_layers),
    )

    LOGGER.info(f"Binary classifier RTL generation complete! Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
