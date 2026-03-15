#!/usr/bin/env python3
"""
Minimal rtl_mapper for binary_onnx_to_rtl.py.
Contains only the functions needed for ONNX-to-RTL conversion (no PyTorch model loading).
Output format matches binaryclass_nn: PyramidTech header, clk_i/rst_n_i, _s/_q suffixes, etc.
Uses reusable templates from binaryclass_nn/ (without headers) when available.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Reusable Templates
# -----------------------------------------------------------------------------

_RTL_MAPPER_DIR = Path(__file__).resolve().parent
BINARYCLASS_NN_TEMPLATES_DIR = _RTL_MAPPER_DIR / "binaryclass_nn"

# Reusable template filenames (used without changes, headers stripped)
_REUSABLE_TEMPLATES = (
    "mac.sv",
    "fc_in.sv",
    "fc_out.sv",
    "relu_layer.sv",
    "sigmoid_layer.sv",
    "sync_fifo.sv",
    "quant_pkg.sv",
)


def _strip_pyramidtech_header(content: str) -> str:
    """Remove PyramidTech header block. Return RTL content from first `begin_keywords or package."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("`") or (s.startswith("package") and "quant_pkg" in s):
            result = "\n".join(lines[i:])
            return result + "\n" if content.endswith("\n") else result
    return content


def _load_reusable_template(filename: str) -> Optional[str]:
    """Load template from binaryclass_nn/ and strip header. Returns None if file not found."""
    path = BINARYCLASS_NN_TEMPLATES_DIR / filename
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return _strip_pyramidtech_header(content)
    except Exception as e:
        LOGGER.warning(f"Could not load template {filename}: {e}")
        return None


def _get_quant_pkg_from_template(weight_width: int) -> str:
    """Get quant_pkg content: use binaryclass_nn template + Q_INT define, or fallback to embedded."""
    template = _load_reusable_template("quant_pkg.sv")
    if template:
        return f"`define Q_INT{weight_width}\n" + template
    return _get_quant_pkg_content(weight_width)


def _write_template_or_embedded(sv_dir: Path, filename: str, template_content: Optional[str], embedded_content: str) -> None:
    """Write template from binaryclass_nn (no header) if available, else embedded."""
    content = template_content if template_content else _pyramidtech_wrap(embedded_content, filename, "")
    (sv_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Formatting Helpers (q_data declarations)
# -----------------------------------------------------------------------------

def _pyramidtech_wrap(content: str, file_name: str, description: str) -> str:
    """Return RTL content as-is (no header/keywords wrapper)."""
    return content.rstrip()


def _rtl_header(filename: str, module_name: str, description: str) -> str:
    """Generate file header block matching binaryclass_nn style."""
    lines = description.strip().split("\n")
    desc_block = "\n".join(f"//   {line.strip()}" for line in lines if line.strip())
    return f"""//==============================================================================
// File      : {filename}
// Module    : {module_name}
// Description:
{desc_block}
//==============================================================================

"""


def _q_data_decl(name: str, size: int, prefix: str = "    ") -> str:
    """Format q_data_t declaration: scalar when size==1, array otherwise."""
    if size == 1:
        return f"{prefix}q_data_t   {name};"
    return f"{prefix}q_data_t   {name} [{size}];"


def _q_data_port_decl(name: str, size: int, direction: str = "output") -> str:
    """Format q_data_t port declaration: scalar when size==1, array otherwise."""
    if size == 1:
        return f"    {direction} q_data_t {name}"
    return f"    {direction} q_data_t {name} [{size}]"


# -----------------------------------------------------------------------------
# Quantization and Layer Data Types
# -----------------------------------------------------------------------------

def float_to_int(val: np.ndarray, scale: int, bit_width: int) -> np.ndarray:
    """Quantize float array to signed integer with specified bit width."""
    if bit_width == 4:
        min_val, max_val = -8, 7
        dtype = np.int8
    elif bit_width == 8:
        min_val, max_val = -128, 127
        dtype = np.int8
    elif bit_width == 16:
        min_val, max_val = -32768, 32767
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported bit width: {bit_width}. Supported: 4, 8, 16")

    quantized = np.clip(np.round(val.astype(np.float32) * float(scale)), min_val, max_val)
    return quantized.astype(dtype)


@dataclass
class LayerInfo:
    """Information about a layer in the model."""
    name: str
    layer_type: str  # 'flatten', 'linear', 'relu'
    module_qualname: Optional[str] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None
    quant_params: Optional[Dict[str, Any]] = None  # ONNX: weight_scale, b_zero_point, etc.
    activation: Optional[str] = None  # ONNX: Relu, Sigmoid, Tanh, etc. (from graph trace)


# -----------------------------------------------------------------------------
# Memory File Generation (.mem format for weights and biases)
# -----------------------------------------------------------------------------

def generate_quant_pkg_style_weight_mem(
    weight_matrix: np.ndarray,
    out_path: Path,
    layer_name: str,
    in_features: int,
    out_features: int,
    bit_width: int
) -> None:
    """Generate weight .mem in quant_pkg format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weight_matrix = np.asarray(weight_matrix, dtype=np.int32)
    mask = (1 << bit_width) - 1

    with out_path.open("w", encoding="utf-8") as wf:
        if layer_name == "fc1":
            num_neurons = out_features
            total_bits = num_neurons * bit_width
            hex_width = (total_bits + 3) // 4
            for j in range(in_features):
                packed = 0
                for neuron_idx in range(num_neurons):
                    val = int(weight_matrix[neuron_idx, j]) & mask
                    packed |= val << (neuron_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")
        else:
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = (total_bits + 3) // 4
            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")


def generate_quant_pkg_style_bias_mem(
    bias_vector: np.ndarray,
    out_path: Path,
    num_neurons: int,
    bit_width: int
) -> None:
    """Generate bias .mem in quant_pkg format: single line, all biases packed (neuron 0 in LSBs)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)
    mask = (1 << bit_width) - 1
    total_bits = num_neurons * bit_width
    hex_width = (total_bits + 3) // 4
    packed = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        packed |= val << (neuron_idx * bit_width)
    with out_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed:0{hex_width}X}\n")


def generate_proj_bias_mem_acc(
    bias_vector: np.ndarray,
    out_path: Path,
    num_neurons: int,
    bit_width: int,
) -> None:
    """Generate bias .mem for fc_in (acc_t packed): one row, each bias sign-extended to 4*bit_width."""
    acc_width = 4 * bit_width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)
    mask = (1 << bit_width) - 1
    total_bits = num_neurons * acc_width
    hex_width = (total_bits + 3) // 4
    packed = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        if val >= (1 << (bit_width - 1)):
            val = val - (1 << bit_width)
        val_acc = val & ((1 << acc_width) - 1)
        packed |= val_acc << (neuron_idx * acc_width)
    with out_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed:0{hex_width}X}\n")


def generate_proj_out_bias_mem(
    bias_vector: np.ndarray,
    out_path: Path,
    bit_width: int,
) -> None:
    """Generate bias .mem for fc_out: one row per bias, each as acc_t (4*bit_width bits)."""
    acc_width = 4 * bit_width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)
    mask = (1 << bit_width) - 1
    hex_width = (acc_width + 3) // 4
    with out_path.open("w", encoding="utf-8") as bf:
        for val in bias_vector:
            val = int(val) & mask
            if val >= (1 << (bit_width - 1)):
                val = val - (1 << bit_width)
            val_acc = val & ((1 << acc_width) - 1)
            bf.write(f"{val_acc:0{hex_width}X}\n")


def generate_proj_mem_files(
    layers: List[LayerInfo],
    mem_dir: Path,
    scale: int,
    bit_width: int,
    *,
    binaryclass_nn_layout: bool = False,
    six_layer_blocks: bool = False,
) -> None:
    """Generate .mem files with fc_N_proj_in/out naming (binaryclass_nn style per layer block).
    Pairs layers consecutively: block 0=(fc1,fc2), block 1=(fc3,fc4), ... For odd N, last block=(fcN, 1x1).
    six_layer_blocks is deprecated; pairing is now automatic for any N."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    mem_dir.mkdir(parents=True, exist_ok=True)
    # Pair consecutively: (fc1,fc2), (fc3,fc4), ...; odd N: last block has (fcN, None) -> 1x1 proj_out
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    block_indices = [
        (2 * b, 2 * b + 1) if 2 * b + 1 < n else (2 * b, None)
        for b in range(num_blocks)
    ]
    for block_idx, (layer_idx, next_idx) in enumerate(block_indices):
        layer = linear_layers[layer_idx]
        next_layer = linear_layers[next_idx] if next_idx is not None else None
        prefix = _proj_prefix(block_idx)
        w_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        b_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((layer.out_features or 0,), dtype=np.float32)
        wq = float_to_int(w_np, scale, bit_width)
        bq = float_to_int(b_np, scale, bit_width)
        in_f = layer.in_features or 0
        out_f = layer.out_features or 0

        # proj_in: weights + bias (per layer block, as in binaryclass_nn)
        generate_quant_pkg_style_weight_mem(wq, mem_dir / f"{prefix}_in_weights.mem", "fc1", in_f, out_f, bit_width)
        generate_proj_bias_mem_acc(bq, mem_dir / f"{prefix}_in_bias.mem", out_f, bit_width)

        # proj_out: next layer weights + bias (per layer block, as in binaryclass_nn)
        if next_layer is not None:
            next_w = next_layer.weight.detach().cpu().numpy().astype(np.float32)
            next_b = next_layer.bias.detach().cpu().numpy().astype(np.float32) if next_layer.bias is not None else np.zeros((next_layer.out_features or 0,), dtype=np.float32)
            next_wq = float_to_int(next_w, scale, bit_width)
            next_bq = float_to_int(next_b, scale, bit_width)
            next_in = next_layer.in_features or 0
            next_out = next_layer.out_features or 0
            generate_quant_pkg_style_weight_mem(next_wq, mem_dir / f"{prefix}_out_weights.mem", "fc2", next_in, next_out, bit_width)
            generate_proj_out_bias_mem(next_bq, mem_dir / f"{prefix}_out_bias.mem", bit_width)
        else:
            # Last layer: proj_out is 1x1 (binary output, as in binaryclass_nn)
            next_out = 1
            w_out = float_to_int(np.eye(1, out_f, dtype=np.float32) * (scale / 256.0), scale, bit_width) if out_f == 1 else float_to_int(np.zeros((1, out_f), dtype=np.float32), scale, bit_width)
            bq_out = float_to_int(np.array([float(bq[0])] if out_f >= 1 else [0.0], dtype=np.float32), scale, bit_width)
            generate_quant_pkg_style_weight_mem(w_out, mem_dir / f"{prefix}_out_weights.mem", "fc2", out_f, next_out, bit_width)
            generate_proj_out_bias_mem(bq_out, mem_dir / f"{prefix}_out_bias.mem", bit_width)


def generate_quant_pkg_style_mems(
    layer_name: str,
    weight_matrix: np.ndarray,
    bias_vector: np.ndarray,
    bit_width: int,
    rtl_root: Path,
) -> None:
    """Generate mem files for legacy flow."""
    if bit_width not in (4, 8, 16):
        raise ValueError(f"Unsupported bit width: {bit_width}. Supported: 4, 8, 16")

    mem_dir = rtl_root / "mem"
    mem_dir.mkdir(parents=True, exist_ok=True)

    weight_matrix = np.asarray(weight_matrix, dtype=np.int32)
    bias_vector = np.asarray(bias_vector, dtype=np.int32)

    out_features, in_features = weight_matrix.shape
    mask = (1 << bit_width) - 1

    weights_path = mem_dir / f"{layer_name}_weights_{bit_width}.mem"
    with weights_path.open("w", encoding="utf-8") as wf:
        if layer_name == "fc1":
            num_neurons = out_features
            total_bits = num_neurons * bit_width
            hex_width = total_bits // 4
            for j in range(in_features):
                packed = 0
                for neuron_idx in range(num_neurons):
                    val = int(weight_matrix[neuron_idx, j]) & mask
                    packed |= val << (neuron_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")
        else:
            num_inputs = in_features
            total_bits = num_inputs * bit_width
            hex_width = total_bits // 4
            for neuron_idx in range(out_features):
                packed = 0
                for inp_idx in range(num_inputs):
                    val = int(weight_matrix[neuron_idx, inp_idx]) & mask
                    packed |= val << (inp_idx * bit_width)
                wf.write(f"{packed:0{hex_width}X}\n")

    biases_path = mem_dir / f"{layer_name}_biases_{bit_width}.mem"
    num_neurons = bias_vector.shape[0]
    total_bits = num_neurons * bit_width
    hex_width = (total_bits + 3) // 4
    packed_biases = 0
    for neuron_idx in range(num_neurons):
        val = int(bias_vector[neuron_idx]) & mask
        packed_biases |= val << (neuron_idx * bit_width)
    with biases_path.open("w", encoding="utf-8") as bf:
        bf.write(f"{packed_biases:0{hex_width}X}\n")


def emit_legacy_rtl_outputs(
    legacy_rtl_dir: Path,
    layers: List[LayerInfo],
    scale: int,
    bits_list: Sequence[int] = (4, 8, 16),
    write_sv: bool = True,
) -> None:
    """Emit legacy rtl/ flow outputs (mem files only; write_sv not supported)."""
    legacy_rtl_dir = legacy_rtl_dir.resolve()

    if write_sv:
        LOGGER.warning("Legacy SV file writing not supported in minimal rtl_mapper; emitting mem files only.")

    for layer in layers:
        if layer.layer_type != "linear":
            continue

        if layer.weight is None:
            raise RuntimeError(f"Linear layer {layer.name} missing weights")

        weight_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        bias_np = (
            layer.bias.detach().cpu().numpy().astype(np.float32)
            if layer.bias is not None
            else np.zeros((weight_np.shape[0],), dtype=np.float32)
        )

        for bits in bits_list:
            wq = float_to_int(weight_np, scale, bits)
            bq = float_to_int(bias_np, scale, bits)
            generate_quant_pkg_style_mems(layer.name, wq, bq, bits, legacy_rtl_dir)


# -----------------------------------------------------------------------------
# Embedded RTL Templates (quant_pkg, mac, fc_in, fc_out, relu_layer)
# -----------------------------------------------------------------------------

def _get_quant_pkg_content(weight_width: int) -> str:
    """Generate quant_pkg.sv matching binaryclass_nn reference."""
    if weight_width not in (4, 8, 16):
        weight_width = 16
    define_line = f"`define Q_INT{weight_width}\n"
    default_def = "`ifndef Q_INT4\n`ifndef Q_INT8\n`ifndef Q_INT16\n`define Q_INT16\n`endif\n`endif\n`endif\n"
    body = define_line + default_def + '''package quant_pkg;

   // =============================================================
   // Quantization mode (select ONE at compile time)
   // =============================================================
   `ifdef Q_INT4
      localparam int Q_WIDTH = 4;
   `elsif Q_INT8
      localparam int Q_WIDTH = 8;
   `elsif Q_INT16
      localparam int Q_WIDTH = 16;
   `else
      localparam int Q_WIDTH = 8;
   `endif

   // =============================================================
   // Common fixed-point types
   // =============================================================
   typedef logic signed [Q_WIDTH-1:0]       q_data_t;
   typedef logic signed [2*Q_WIDTH-1:0]     q_mult_t;
   typedef logic signed [4*Q_WIDTH-1:0]     acc_t;

   // =============================================================
   // Method A: Narrow limits (Q-width), then cast to acc_t
   // =============================================================
   localparam q_data_t Q_MAX = {1'b0, {Q_WIDTH-1{1'b1}}};
   localparam q_data_t Q_MIN = {1'b1, {Q_WIDTH-1{1'b0}}};

   localparam acc_t ACC_Q_MAX = acc_t'(Q_MAX);
   localparam acc_t ACC_Q_MIN = acc_t'(Q_MIN);

   // =============================================================
   // Method B: Native accumulator limits (full acc_t width)
   // =============================================================
   localparam acc_t ACC_FULL_MAX = {1'b0, {4*Q_WIDTH-1{1'b1}}};
   localparam acc_t ACC_FULL_MIN = {1'b1, {4*Q_WIDTH-1{1'b0}}};

   localparam q_data_t SIGMOID_MAX = 2**(Q_WIDTH-2);
   localparam q_data_t SIGMOID_MIN = 0;

endpackage
'''
    return _pyramidtech_wrap(body, "quant_pkg.sv", "Package defining quantization widths, fixed-point data types, and saturation limits.")


EMBEDDED_MAC_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module mac (
   input  logic clk,
   input  logic rst_n,
   input  logic enable,
   input  logic clear,

   input  q_data_t a,
   input  q_data_t b,

   output acc_t acc,
   output logic valid_out
);

   // ------------------------------------------------------------
   // Internal signals
   // ------------------------------------------------------------
   q_mult_t mult_reg;
   acc_t    acc_reg;

   logic enable_q;
   logic enable_qq;

   // ------------------------------------------------------------
   // Stage 1: Multiply
   // ------------------------------------------------------------
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         mult_reg <= '0;
      else if (clear)
         mult_reg <= '0;
      else if (enable)
         mult_reg <= $signed(a) * $signed(b);
   end

   // ------------------------------------------------------------
   // Stage 2: Accumulate
   // ------------------------------------------------------------
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         acc_reg <= '0;
      else if (clear)
         acc_reg <= '0;
      else if (enable_q)
         acc_reg <= acc_reg + mult_reg;
   end

   // ------------------------------------------------------------
   // Stage 3: Output saturation to ACC full range
   // ------------------------------------------------------------
   always_ff @(posedge clk) begin
      if (acc_reg > $signed(ACC_FULL_MAX))
         acc <= ACC_FULL_MAX;
      else if (acc_reg < $signed(ACC_FULL_MIN))
         acc <= ACC_FULL_MIN;
      else
         acc <= acc_reg[4*Q_WIDTH-1:0];
   end

   // ------------------------------------------------------------
   // Valid signal pipeline (matches MAC latency)
   // ------------------------------------------------------------
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         valid_out <= 1'b0;
      end
      else if (clear) begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         valid_out <= 1'b0;
      end
      else begin
         enable_q  <= enable;
         enable_qq <= enable_q;
         valid_out <= enable_qq;
      end
   end

endmodule
'''


EMBEDDED_FC_IN_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_in #(
   parameter int NUM_NEURONS   = 8,
   parameter int INPUT_SIZE    = 16,
   parameter signed BIAS_SCALE  = 0,
   parameter signed LAYER_SCALE = 12
)(
   input  logic        clk,
   input  logic        rst_n,
   input  logic        valid_in,
   input  q_data_t     data_in,
   input  q_data_t     weights[NUM_NEURONS],
   input  acc_t        biases[NUM_NEURONS],

   output q_data_t     data_out[NUM_NEURONS],
   output logic        valid_out
);

   // ------------------------------------------------------------
   // Internal MAC signals
   // ------------------------------------------------------------
   acc_t mac_acc[NUM_NEURONS];
   acc_t acc_tmp[NUM_NEURONS];
   acc_t bias_aligned[NUM_NEURONS];
   acc_t data_out_temp[NUM_NEURONS];

   logic [$clog2(INPUT_SIZE+1)-1:0] count_out;

   logic mac_enable;
   logic mac_clear;

   logic [NUM_NEURONS-1:0] mac_valid_out;
   logic all_mac_valid;

   // ------------------------------------------------------------
   // Instantiate MACs
   // ------------------------------------------------------------
   genvar i;
   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_MACS
         mac u_mac (
            .clk(clk),
            .rst_n(rst_n),
            .enable(mac_enable),
            .clear(mac_clear),
            .a(data_in),
            .b(weights[i]),
            .valid_out(mac_valid_out[i]),
            .acc(mac_acc[i])
         );
      end
   endgenerate

   assign all_mac_valid = &mac_valid_out;
   assign mac_enable = valid_in;
   assign mac_clear  = valid_out;

   // ------------------------------------------------------------
   // Count valid inputs and generate valid_out for last input
   // ------------------------------------------------------------
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         count_out <= '0;
         valid_out <= 1'b0;
      end else begin
         valid_out <= 1'b0;
         if (all_mac_valid) begin
            if (count_out == INPUT_SIZE-1) begin
               count_out <= '0;
               valid_out <= 1'b1;
            end else begin
               count_out <= count_out + 1'b1;
            end
         end
      end
   end

   // ------------------------------------------------------------
   // Bias addition, layer scaling, and output saturation
   // ------------------------------------------------------------
   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_BIAS_ADD
         assign bias_aligned[i] = (BIAS_SCALE >= 0) ? (biases[i] <<< BIAS_SCALE) : (biases[i] >>> -BIAS_SCALE);
         assign acc_tmp[i] = mac_acc[i] + bias_aligned[i];
         assign data_out_temp[i] = (LAYER_SCALE >= 0) ? (acc_tmp[i] >>> LAYER_SCALE) : (acc_tmp[i] <<< -LAYER_SCALE);
         always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
               data_out[i] <= '0;
            else if (count_out == INPUT_SIZE-1) begin
               if (data_out_temp[i] > ACC_Q_MAX)
                  data_out[i] <= Q_MAX;
               else if (data_out_temp[i] < ACC_Q_MIN)
                  data_out[i] <= Q_MIN;
               else
                  data_out[i] <= data_out_temp[i][Q_WIDTH-1:0];
            end
         end
      end
   endgenerate

endmodule
'''


EMBEDDED_FC_OUT_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_out #(
   parameter int      NUM_NEURONS = 2,
   parameter signed   LAYER_SCALE = 5,
   parameter signed   BIAS_SCALE  = 1
)(
   input  logic clk,
   input  logic rst_n,
   input  logic valid_in,

   input  q_data_t data_in[NUM_NEURONS],
   input  q_data_t weights[NUM_NEURONS],
   input  acc_t    bias,

   output q_data_t data_out,
   output logic    valid_out
);

   // ============================================================
   // Stage 0: Multipliers (combinational)
   // ============================================================
   acc_t mult_res[NUM_NEURONS];
   acc_t bias_pipe;

   genvar i;
   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : GEN_MULT
         assign mult_res[i] = $signed(data_in[i]) * $signed(weights[i]);
      end
   endgenerate

   // ============================================================
   // Stage 1: Register multiplier outputs
   // ============================================================
   acc_t mult_pipe[NUM_NEURONS];
   logic valid_p1;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         for (int j = 0; j < NUM_NEURONS; j++)
            mult_pipe[j] <= '0;
         valid_p1 <= 1'b0;
         bias_pipe <= '0;
      end else begin
         for (int j = 0; j < NUM_NEURONS; j++)
            mult_pipe[j] <= mult_res[j];
         valid_p1 <= valid_in;
         bias_pipe <= bias;
      end
   end

   // ============================================================
   // Stage 2: Adder + Bias + Layer scaling (combinational)
   // ============================================================
   acc_t sum_stage2;
   acc_t sum_stage2_tmp;

   always_comb begin
      if (BIAS_SCALE >= 0)
         sum_stage2_tmp = bias_pipe <<< BIAS_SCALE;
      else
         sum_stage2_tmp = bias_pipe >>> -BIAS_SCALE;
      for (int j = 0; j < NUM_NEURONS; j++)
         sum_stage2_tmp += mult_pipe[j];
      if (LAYER_SCALE >= 0)
         sum_stage2 = sum_stage2_tmp >>> LAYER_SCALE;
      else
         sum_stage2 = sum_stage2_tmp <<< -LAYER_SCALE;
   end

   // ============================================================
   // Stage 2 Register
   // ============================================================
   acc_t sum_pipe2;
   logic valid_p2;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         sum_pipe2 <= '0;
         valid_p2  <= 1'b0;
      end else begin
         sum_pipe2 <= sum_stage2;
         valid_p2  <= valid_p1;
      end
   end

   // ============================================================
   // Stage 3: Saturation + Output Register
   // ============================================================
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         data_out  <= '0;
         valid_out <= 1'b0;
      end else begin
         valid_out <= valid_p2;
         if (sum_pipe2 > ACC_Q_MAX)
            data_out <= Q_MAX;
         else if (sum_pipe2 < ACC_Q_MIN)
            data_out <= Q_MIN;
         else
            data_out <= sum_pipe2[Q_WIDTH-1:0];
      end
   end

endmodule
'''


EMBEDDED_RELU_LAYER_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module relu_layer (
   input  logic clk,
   input  logic rst_n,
   input  logic valid_in,
   input  q_data_t data_in,

   output q_data_t data_out,
   output logic valid_out
);

   // ============================================================
   // ReLU logic with pipelined valid signal
   // ============================================================
   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         data_out  <= '0;
         valid_out <= 1'b0;
      end else begin
         valid_out <= valid_in;
         if (valid_in)
            data_out <= (data_in < 0) ? '0 : data_in;
      end
   end

endmodule
'''

# Legacy hierarchical format: relu for array (element-wise)
EMBEDDED_RELU_LAYER_ARRAY_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module relu_layer_array #(
   parameter int NUM_NEURONS = 8
)(
   input  logic clk,
   input  logic rst_n,
   input  logic valid_in,
   input  q_data_t data_in[NUM_NEURONS],

   output q_data_t data_out[NUM_NEURONS],
   output logic valid_out
);

   integer i;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         valid_out <= 1'b0;
      else
         valid_out <= valid_in;
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         for (i = 0; i < NUM_NEURONS; i++)
            data_out[i] <= '0;
      end else if (valid_in) begin
         for (i = 0; i < NUM_NEURONS; i++)
            data_out[i] <= (data_in[i] < 0) ? '0 : data_in[i];
      end
   end

endmodule
'''


# -----------------------------------------------------------------------------
# Embedded binaryclass_nn format templates (AXI4-Stream, reference structure)
# -----------------------------------------------------------------------------

BINARYCLASS_QUANT_PKG_SV = '''package quant_pkg;

   `ifdef Q_INT4
      localparam int Q_WIDTH = 4;
   `elsif Q_INT8
      localparam int Q_WIDTH = 8;
   `elsif Q_INT16
      localparam int Q_WIDTH = 16;
   `else
      localparam int Q_WIDTH = 8;
   `endif

   typedef logic signed [Q_WIDTH-1:0]   q_data_t;
   typedef logic signed [2*Q_WIDTH-1:0] q_mult_t;
   typedef logic signed [4*Q_WIDTH-1:0] acc_t;

   localparam q_data_t Q_MAX = {{1'b0, {{Q_WIDTH-1{{1'b1}}}}}};
   localparam q_data_t Q_MIN = {{1'b1, {{Q_WIDTH-1{{1'b0}}}}}};

   localparam acc_t ACC_Q_MAX = acc_t'(Q_MAX);
   localparam acc_t ACC_Q_MIN = acc_t'(Q_MIN);

   localparam acc_t ACC_FULL_MAX = {{1'b0, {{4*Q_WIDTH-1{{1'b1}}}}}};
   localparam acc_t ACC_FULL_MIN = {{1'b1, {{4*Q_WIDTH-1{{1'b0}}}}}};

   localparam q_data_t SIGMOID_MAX = 2**(Q_WIDTH-2);
   localparam q_data_t SIGMOID_MIN = 0;

endpackage
'''

BINARYCLASS_MAC_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module mac (
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic enable_i,
    input  logic clear_i,
    input  q_data_t a_i,
    input  q_data_t b_i,
    output acc_t acc_o,
    output logic valid_o
);

    q_mult_t mult_reg;
    acc_t acc_reg;
    logic enable_q;
    logic enable_qq;

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            mult_reg <= '0;
        else if (clear_i)
            mult_reg <= '0;
        else if (enable_i)
            mult_reg <= $signed(a_i) * $signed(b_i);
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            acc_reg <= '0;
        else if (clear_i)
            acc_reg <= '0;
        else if (enable_q)
            acc_reg <= acc_reg + mult_reg;
    end

    always_ff @(posedge clk_i) begin
        if (acc_reg > $signed(ACC_FULL_MAX))
            acc_o <= ACC_FULL_MAX;
        else if (acc_reg < $signed(ACC_FULL_MIN))
            acc_o <= ACC_FULL_MIN;
        else
            acc_o <= acc_reg[4*Q_WIDTH-1:0];
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            enable_q  <= 1'b0;
            enable_qq <= 1'b0;
            valid_o   <= 1'b0;
        end else if (clear_i) begin
            enable_q  <= 1'b0;
            enable_qq <= 1'b0;
            valid_o   <= 1'b0;
        end else begin
            enable_q  <= enable_i;
            enable_qq <= enable_q;
            valid_o   <= enable_qq;
        end
    end

endmodule
'''

BINARYCLASS_FC_IN_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_in #(
    parameter int NUM_NEURONS   = 8,
    parameter int INPUT_SIZE    = 16,
    parameter signed BIAS_SCALE  = 0,
    parameter signed LAYER_SCALE = 12
)(
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i,
    input  q_data_t weights_i[NUM_NEURONS],
    input  acc_t biases_i[NUM_NEURONS],
    output q_data_t data_o[NUM_NEURONS],
    output logic valid_o
);

    acc_t mac_acc[NUM_NEURONS];
    acc_t acc_tmp[NUM_NEURONS];
    acc_t bias_aligned[NUM_NEURONS];
    acc_t data_out_temp[NUM_NEURONS];
    logic [$clog2(INPUT_SIZE+1)-1:0] count_out;
    logic mac_enable;
    logic mac_clear;
    logic [NUM_NEURONS-1:0] mac_valid_out;
    logic all_mac_valid;

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : FC_MACS
            mac u_mac (
                .clk_i(clk_i),
                .rst_n_i(rst_n_i),
                .enable_i(mac_enable),
                .clear_i(mac_clear),
                .a_i(data_i),
                .b_i(weights_i[i]),
                .valid_o(mac_valid_out[i]),
                .acc_o(mac_acc[i])
            );
        end
    endgenerate

    assign all_mac_valid = &mac_valid_out;
    assign mac_enable = valid_i;
    assign mac_clear  = valid_o;

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            count_out <= '0;
            valid_o   <= 1'b0;
        end else begin
            valid_o <= 1'b0;
            if (all_mac_valid) begin
                if (count_out == INPUT_SIZE-1) begin
                    count_out <= 0;
                    valid_o   <= 1'b1;
                end else begin
                    count_out <= count_out + 1'b1;
                end
            end
        end
    end

    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : FC_BIAS_ADD
            assign bias_aligned[i] = (BIAS_SCALE >= 0) ? (biases_i[i] <<< BIAS_SCALE) : (biases_i[i] >>> -BIAS_SCALE);
            assign acc_tmp[i] = mac_acc[i] + bias_aligned[i];
            assign data_out_temp[i] = (LAYER_SCALE >= 0) ? (acc_tmp[i] >>> LAYER_SCALE) : (acc_tmp[i] <<< -LAYER_SCALE);
            always_ff @(posedge clk_i or negedge rst_n_i) begin
                if (!rst_n_i)
                    data_o[i] <= '0;
                else if (count_out == INPUT_SIZE-1) begin
                    if (data_out_temp[i] > ACC_Q_MAX)
                        data_o[i] <= Q_MAX;
                    else if (data_out_temp[i] < ACC_Q_MIN)
                        data_o[i] <= Q_MIN;
                    else
                        data_o[i] <= data_out_temp[i][Q_WIDTH-1:0];
                end
            end
        end
    endgenerate

endmodule
'''

BINARYCLASS_FC_OUT_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_out #(
    parameter int NUM_NEURONS = 2,
    parameter signed LAYER_SCALE = 5,
    parameter signed BIAS_SCALE  = 1
)(
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i[NUM_NEURONS],
    input  q_data_t weights_i[NUM_NEURONS],
    input  acc_t bias_i,
    output q_data_t data_o,
    output logic valid_o
);

    acc_t mult_res[NUM_NEURONS];
    acc_t bias_pipe;
    acc_t mult_pipe[NUM_NEURONS];
    logic valid_p1;
    acc_t sum_stage2;
    acc_t sum_stage2_tmp;
    acc_t sum_pipe2;
    logic valid_p2;

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : GEN_MULT
            assign mult_res[i] = $signed(data_i[i]) * $signed(weights_i[i]);
        end
    endgenerate

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (int j = 0; j < NUM_NEURONS; j++)
                mult_pipe[j] <= '0;
            valid_p1 <= 1'b0;
            bias_pipe <= '0;
        end else begin
            for (int j = 0; j < NUM_NEURONS; j++)
                mult_pipe[j] <= mult_res[j];
            valid_p1 <= valid_i;
            bias_pipe <= bias_i;
        end
    end

    always_comb begin
        if (BIAS_SCALE >= 0)
            sum_stage2_tmp = bias_pipe <<< BIAS_SCALE;
        else
            sum_stage2_tmp = bias_pipe >>> -BIAS_SCALE;
        for (int j = 0; j < NUM_NEURONS; j++)
            sum_stage2_tmp += mult_pipe[j];
        if (LAYER_SCALE >= 0)
            sum_stage2 = sum_stage2_tmp >>> LAYER_SCALE;
        else
            sum_stage2 = sum_stage2_tmp <<< -LAYER_SCALE;
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            sum_pipe2 <= '0;
            valid_p2  <= 1'b0;
        end else begin
            sum_pipe2 <= sum_stage2;
            valid_p2  <= valid_p1;
        end
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            data_o  <= '0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_p2;
            if (sum_pipe2 > ACC_Q_MAX)
                data_o <= Q_MAX;
            else if (sum_pipe2 < ACC_Q_MIN)
                data_o <= Q_MIN;
            else
                data_o <= sum_pipe2[Q_WIDTH-1:0];
        end
    end

endmodule
'''

BINARYCLASS_RELU_LAYER_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module relu_layer (
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i,
    output q_data_t data_o,
    output logic valid_o
);

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            data_o  <= 8'd0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_i;
            if (valid_i)
                data_o <= (data_i < 0) ? 8'd0 : data_i;
        end
    end

endmodule
'''

BINARYCLASS_SIGMOID_LAYER_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module sigmoid_layer (
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i,
    output q_data_t data_o,
    output logic valid_o
);

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            data_o  <= 8'd0;
            valid_o <= 1'b0;
        end else begin
            valid_o <= valid_i;
            if (valid_i) begin
                if (data_i >= 0)
                    data_o <= SIGMOID_MAX;
                else
                    data_o <= SIGMOID_MIN;
            end
        end
    end

endmodule
'''

BINARYCLASS_SYNC_FIFO_SV = '''`timescale 1ns/1ps

module sync_fifo #(
  parameter int DATA_WIDTH = 32,
  parameter int DEPTH      = 1024
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  logic                  write_en_i,
  input  logic [DATA_WIDTH-1:0] write_data_i,
  output logic                  full_o,

  input  logic                  read_en_i,
  output logic [DATA_WIDTH-1:0] read_data_o,
  output logic                  empty_o
);

  timeunit 1ns;
  timeprecision 1ps;

  localparam int ADDR_WIDTH = $clog2(DEPTH);

  logic [DATA_WIDTH-1:0] fifo_mem_q [0:DEPTH-1];
  logic [ADDR_WIDTH:0]   write_ptr_q;
  logic [ADDR_WIDTH:0]   read_ptr_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : write_logic
    if (!rst_n_i) write_ptr_q <= '0;
    else if (write_en_i && !full_o) begin
      fifo_mem_q[write_ptr_q[ADDR_WIDTH-1:0]] <= write_data_i;
      write_ptr_q <= write_ptr_q + 1'b1;
    end
  end : write_logic

  always_ff @(posedge clk_i or negedge rst_n_i) begin : read_logic
    if (!rst_n_i) read_ptr_q <= '0;
    else if (read_en_i && !empty_o) read_ptr_q <= read_ptr_q + 1'b1;
  end : read_logic

  assign read_data_o = fifo_mem_q[read_ptr_q[ADDR_WIDTH-1:0]];

  assign empty_o = (write_ptr_q == read_ptr_q);
  assign full_o  = (write_ptr_q[ADDR_WIDTH] != read_ptr_q[ADDR_WIDTH]) &&
                   (write_ptr_q[ADDR_WIDTH-1:0] == read_ptr_q[ADDR_WIDTH-1:0]);

endmodule : sync_fifo
'''

# Embedded binaryclass_NN top module (default structure, params substituted from ONNX)
BINARYCLASS_NN_SV = r'''`begin_keywords "1800-2012"
module binaryclass_NN 
  import quant_pkg::*;
#(
  parameter int FC_1_NEURONS          = 32'd30,  
  parameter int FC_2_NEURONS          = 32'd5,   
  parameter int FC_3_NEURONS          = 32'd1,   
  parameter int FC_1_INPUT_SIZE       = 32'd720, 
  parameter int FC_2_INPUT_SIZE       = 32'd45,  
  parameter int FC_3_INPUT_SIZE       = 32'd5,   
  parameter int FC_1_ROM_DEPTH        = 32'd45,  
  parameter int FC_2_ROM_DEPTH        = 32'd5,   
  parameter int FC_3_ROM_DEPTH        = 32'd1, 
  parameter int FC_1_IN_LAYER_SCALE   = 32'd12,
  parameter int FC_1_IN_BIAS_SCALE    = 32'd0,
  parameter int FC_1_OUT_LAYER_SCALE  = 32'd7,
  parameter int FC_1_OUT_BIAS_SCALE   = 32'd0,
  parameter int FC_2_IN_LAYER_SCALE   = 32'd7,
  parameter int FC_2_IN_BIAS_SCALE    = 32'd0,
  parameter int FC_2_OUT_LAYER_SCALE  = 32'd7,
  parameter int FC_2_OUT_BIAS_SCALE   = 32'd0,
  parameter int FC_3_IN_LAYER_SCALE   = 32'd7,
  parameter int FC_3_IN_BIAS_SCALE    = 32'd0,
  parameter int FC_3_OUT_LAYER_SCALE  = 32'd6,
  parameter int FC_3_OUT_BIAS_SCALE   = 32'd0,
  parameter int FIFO_DEPTH            = 32'd1024  
)(
  input  logic clk_i,
  input  logic rst_n_i,

  // AXI4-Stream Slave Interface: Data applied for prediction
  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  // AXI4-Stream Master Interface: Prediction result data
  output logic [DATA_WIDTH-1:0]       m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0]       m_axis_prediction_tkeep_o,
  output logic                        m_axis_prediction_tvalid_o,
  input  logic                        m_axis_prediction_tready_i,
  output logic                        m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  // -------------------------------------------------------------------------
  // Internal signals (Suffixes: _s = signal/wire, _q = register/flop)
  // -------------------------------------------------------------------------
  
  // Layer 1 signals
  q_data_t fc1_out_s[FC_1_NEURONS];
  logic    fc1_valid_s;
  q_data_t fc1_pre_relu_s;
  q_data_t fc1_post_relu_s;
  logic    relu1_valid_i_s;
  logic    relu1_valid_o_s;

  // Layer 2 signals
  q_data_t fc2_out_s[FC_2_NEURONS];
  logic    fc2_valid_s;
  q_data_t fc2_pre_relu_s;
  q_data_t fc2_post_relu_s;
  logic    relu2_valid_i_s;
  logic    relu2_valid_o_s;

  // Layer 3 signals
  q_data_t fc3_out_s[FC_3_NEURONS];
  logic    fc3_valid_s;
  logic    sigmoid_valid_i_s;
  logic    sigmoid_valid_o_s;
  q_data_t sigmoid_data_s;
  q_data_t logits_s;

  //FIFO signals
  logic fifo_empty_s;
  logic fifo_empty_q;
  logic fifo_full_s;
  logic fifo_write_en_s; 
  logic fifo_read_en_s; 
  logic fifo_read_en_q; 
  logic [DATA_WIDTH-1:0] fifo_read_data_s;
  logic [DATA_WIDTH-1:0] fifo_write_data_s;
  // Registered counter for output stream tracking
  logic [$clog2(FC_3_ROM_DEPTH + 1) - 1:0] out_count_q;

  logic tvalid_q;

  // -------------------------------------------------------------------------
  // Layer 1: Fully Connected -> Sequential ROM -> ReLU Activation
  // -------------------------------------------------------------------------
  fc_in_layer #(
    .NUM_NEURONS (FC_1_NEURONS),
    .INPUT_SIZE  (FC_1_INPUT_SIZE),
    .BIAS_SCALE  (FC_1_IN_BIAS_SCALE),
    .LAYER_SCALE (FC_1_IN_LAYER_SCALE)
  ) u_fc1_in_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (s_axis_tvalid_i),
    .data_i  (s_axis_tdata_i),
    .data_o  (fc1_out_s),
    .valid_o (fc1_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS (FC_1_NEURONS),
    .ROM_DEPTH   (FC_1_ROM_DEPTH),
    .BIAS_SCALE  (FC_1_OUT_BIAS_SCALE),
    .LAYER_SCALE (FC_1_OUT_LAYER_SCALE)
  ) u_fc1_out_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (fc1_valid_s),
    .data_i  (fc1_out_s),
    .data_o  (fc1_pre_relu_s),
    .valid_o (relu1_valid_i_s)
  );
  relu_layer u_relu1 (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (relu1_valid_i_s),
    .data_i  (fc1_pre_relu_s),
    .data_o  (fc1_post_relu_s),
    .valid_o (relu1_valid_o_s)
  );

  // -------------------------------------------------------------------------
  // Layer 2: Fully Connected -> Sequential ROM -> ReLU Activation
  // -------------------------------------------------------------------------
  fc_in_layer #(
    .NUM_NEURONS (FC_2_NEURONS),
    .INPUT_SIZE  (FC_2_INPUT_SIZE),
    .BIAS_SCALE  (FC_2_IN_BIAS_SCALE),
    .LAYER_SCALE (FC_2_IN_LAYER_SCALE)
  ) u_fc2_in_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (relu1_valid_o_s),
    .data_i  (fc1_post_relu_s),
    .data_o  (fc2_out_s),
    .valid_o (fc2_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS (FC_2_NEURONS),
    .ROM_DEPTH   (FC_2_ROM_DEPTH),
    .BIAS_SCALE  (FC_2_OUT_BIAS_SCALE),
    .LAYER_SCALE (FC_2_OUT_LAYER_SCALE)
  ) u_fc2_out_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (fc2_valid_s),
    .data_i  (fc2_out_s),
    .data_o  (fc2_pre_relu_s),
    .valid_o (relu2_valid_i_s)
  );
  relu_layer u_relu2 (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (relu2_valid_i_s),
    .data_i  (fc2_pre_relu_s),
    .data_o  (fc2_post_relu_s),
    .valid_o (relu2_valid_o_s)
  );

  // -------------------------------------------------------------------------
  // Layer 3: Fully Connected -> Output Engine
  // -------------------------------------------------------------------------
  fc_in_layer #(
    .NUM_NEURONS (FC_3_NEURONS),
    .INPUT_SIZE  (FC_3_INPUT_SIZE),
    .BIAS_SCALE  (FC_3_IN_BIAS_SCALE),
    .LAYER_SCALE (FC_3_IN_LAYER_SCALE)
  ) u_fc3_in_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (relu2_valid_o_s),
    .data_i  (fc2_post_relu_s),
    .data_o  (fc3_out_s),
    .valid_o (fc3_valid_s)
  );
  fc_out_layer #(
    .NUM_NEURONS (FC_3_NEURONS),
    .ROM_DEPTH   (FC_3_ROM_DEPTH),
    .BIAS_SCALE  (FC_3_OUT_BIAS_SCALE),
    .LAYER_SCALE (FC_3_OUT_LAYER_SCALE)
  ) u_fc3_out_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (fc3_valid_s),
    .data_i  (fc3_out_s),
    .data_o  (logits_s),
    .valid_o (sigmoid_valid_i_s)
  );

  sigmoid_layer u_sigmoid_layer (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .valid_i (sigmoid_valid_i_s),
    .data_i  (logits_s),
    .data_o  (sigmoid_data_s),
    .valid_o (sigmoid_valid_o_s)
  );

  sync_fifo #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEPTH     (FIFO_DEPTH)     
) u_fifo_sigmoid (
    .clk_i        (clk_i),
    .rst_n_i        (rst_n_i),
    .write_en_i   (fifo_write_en_s),
    .write_data_i (fifo_write_data_s),
    .full_o       (fifo_full_s),
    .read_en_i    (fifo_read_en_s),
    .read_data_o  (fifo_read_data_s),
    .empty_o      (fifo_empty_s)
);

always_ff @(posedge clk_i or negedge rst_n_i) begin : fifo_read_pipe
  if (!rst_n_i)
    fifo_read_en_q <= 1'b0;
  else
    fifo_read_en_q <= fifo_read_en_s;
end : fifo_read_pipe

always_ff @(posedge clk_i or negedge rst_n_i) begin : axi_output_logic
  if (!rst_n_i) begin
    m_axis_prediction_tdata_o <= 32'h0;
    tvalid_q                <= 1'b0;
  end
  else if (fifo_read_en_q) begin
    m_axis_prediction_tdata_o <= fifo_read_data_s;
    tvalid_q                <= 1'b1;
  end
  else if (m_axis_prediction_tready_i) begin
    tvalid_q <= 1'b0;
  end
end : axi_output_logic

always_ff @(posedge clk_i or negedge rst_n_i) begin : output_tracking
  if (!rst_n_i) begin
    out_count_q <= '0;
  end
  else if (m_axis_prediction_tvalid_o && m_axis_prediction_tready_i) begin
    if (out_count_q == (FC_3_ROM_DEPTH - 1))
      out_count_q <= '0;
    else
      out_count_q <= out_count_q + 1'b1;
  end
end : output_tracking

assign fifo_write_en_s   = sigmoid_valid_o_s && !fifo_full_s;
assign fifo_read_en_s    = !fifo_empty_s && m_axis_prediction_tready_i;
assign fifo_write_data_s = {24'h0, sigmoid_data_s};

assign m_axis_prediction_tvalid_o = tvalid_q;
assign m_axis_prediction_tkeep_o  = 4'h1;
assign m_axis_prediction_tlast_o = (m_axis_prediction_tvalid_o && (out_count_q == (FC_3_ROM_DEPTH - 1)));

endmodule : binaryclass_NN
`end_keywords
'''

# Embedded binaryclass_NN_wrapper (params substituted from ONNX)
BINARYCLASS_NN_WRAPPER_SV = r'''`begin_keywords "1800-2012"
module binaryclass_NN_wrapper 
  import quant_pkg::*;
#(
  parameter int FC1_NEURONS    = 35,
  parameter int FC2_NEURONS    = 5,
  parameter int FC3_NEURONS    = 1,
  parameter int FC1_INPUT_SIZE = 720,
  parameter int FC2_INPUT_SIZE = 45,
  parameter int FC3_INPUT_SIZE = 5,
  parameter int FC1_ROM_DEPTH  = 45,
  parameter int FC2_ROM_DEPTH  = 5,
  parameter int FC3_ROM_DEPTH  = 1
)(
  input  logic clk_i,
  input  logic rst_n_i,

  // AXI4-Stream Slave Interface: Data applied for prediction
  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  // AXI4-Stream Master Interface: Prediction result data
  output logic [DATA_WIDTH-1:0]       m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0]       m_axis_prediction_tkeep_o,
  output logic                        m_axis_prediction_tvalid_o,
  input  logic                        m_axis_prediction_tready_i,
  output logic                        m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  binaryclass_NN #(
    .FC_1_NEURONS    (FC1_NEURONS),
    .FC_2_NEURONS    (FC2_NEURONS),
    .FC_3_NEURONS    (FC3_NEURONS),
    .FC_1_INPUT_SIZE (FC1_INPUT_SIZE),
    .FC_2_INPUT_SIZE (FC2_INPUT_SIZE),
    .FC_3_INPUT_SIZE (FC3_INPUT_SIZE),
    .FC_1_ROM_DEPTH  (FC1_ROM_DEPTH),
    .FC_2_ROM_DEPTH  (FC2_ROM_DEPTH),
    .FC_3_ROM_DEPTH  (FC3_ROM_DEPTH)
  ) u_binaryclass_nn (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .s_axis_tdata_i                  (s_axis_tdata_i),
    .s_axis_tvalid_i                 (s_axis_tvalid_i),
    .s_axis_tlast_i                  (s_axis_tlast_i),
    .m_axis_prediction_tdata_o  (m_axis_prediction_tdata_o),
    .m_axis_prediction_tready_i (m_axis_prediction_tready_i),
    .m_axis_prediction_tkeep_o  (m_axis_prediction_tkeep_o),
    .m_axis_prediction_tvalid_o (m_axis_prediction_tvalid_o),
    .m_axis_prediction_tlast_o  (m_axis_prediction_tlast_o)
  );

endmodule : binaryclass_NN_wrapper
`end_keywords
'''


def write_embedded_rtl_templates(sv_dir: Path, weight_width: int, *, write_submodules: bool = True, has_softmax: bool = False) -> None:
    """Write RTL templates: use binaryclass_nn/ (headers stripped) when available, else embedded.
    has_softmax is accepted but ignored (no softmax generation in minimal mapper)."""
    sv_dir.mkdir(parents=True, exist_ok=True)

    # quant_pkg: from template + Q_INT define (detected from ONNX weight_width)
    (sv_dir / "quant_pkg.sv").write_text(_get_quant_pkg_from_template(weight_width), encoding="utf-8")

    if write_submodules:
        # Reusable templates from binaryclass_nn/ (exact line-by-line, no headers)
        _write_template_or_embedded(sv_dir, "mac.sv", _load_reusable_template("mac.sv"), EMBEDDED_MAC_SV)
        _write_template_or_embedded(sv_dir, "fc_in.sv", _load_reusable_template("fc_in.sv"), EMBEDDED_FC_IN_SV)
        _write_template_or_embedded(sv_dir, "fc_out.sv", _load_reusable_template("fc_out.sv"), EMBEDDED_FC_OUT_SV)
        _write_template_or_embedded(sv_dir, "relu_layer.sv", _load_reusable_template("relu_layer.sv"), EMBEDDED_RELU_LAYER_SV)
        _write_template_or_embedded(sv_dir, "sigmoid_layer.sv", _load_reusable_template("sigmoid_layer.sv"), BINARYCLASS_SIGMOID_LAYER_SV)
        _write_template_or_embedded(sv_dir, "sync_fifo.sv", _load_reusable_template("sync_fifo.sv"), BINARYCLASS_SYNC_FIFO_SV)
        (sv_dir / "relu_layer_array.sv").write_text(_pyramidtech_wrap(EMBEDDED_RELU_LAYER_ARRAY_SV, "relu_layer_array.sv", "ReLU activation layer (array, for legacy)."), encoding="utf-8")
        LOGGER.info("Wrote RTL templates (quant_pkg, mac, fc_in, fc_out, relu_layer, sigmoid_layer, sync_fifo from binaryclass_nn or embedded)")
    else:
        LOGGER.info("Wrote quant_pkg.sv only (flattened mode)")


# -----------------------------------------------------------------------------
# Proj ROM naming (fc_proj, fc_2_proj, fc_3_proj, ...)
# -----------------------------------------------------------------------------

def _proj_prefix(layer_idx: int) -> str:
    """Return ROM prefix: fc_proj for layer 0, fc_2_proj for layer 1, fc_3_proj for layer 2, etc."""
    if layer_idx == 0:
        return "fc_proj"
    return f"fc_{layer_idx + 1}_proj"


def generate_proj_roms(
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
    mem_dir: Path,
    *,
    mem_path_prefix: str = "mem_files/",
) -> List[Path]:
    """Generate ROM modules with fc_N_proj_in/out naming for parameterized hierarchical flow.
    mem_path_prefix: path for $readmemh (e.g. 'mem_files/' or '' for binaryclass_nn root layout)."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    paths: List[Path] = []
    for idx, layer in enumerate(linear_layers):
        prefix = _proj_prefix(idx)
        in_f = layer.in_features or 0
        out_f = layer.out_features or 0
        next_in = linear_layers[idx + 1].in_features or 0 if idx + 1 < len(linear_layers) else 1

        # fc_in: weights depth=INPUT_SIZE, bias depth=1
        w_depth_in = in_f
        w_width_in = out_f * weight_width
        b_width_in = out_f * weight_width * 4
        mem_in_w = f"{mem_path_prefix}{prefix}_in_weights.mem"
        mem_in_b = f"{mem_path_prefix}{prefix}_in_bias.mem"

        body_w = f"""`timescale 1ns/1ps

module {prefix}_in_weights_rom #(
    parameter int DEPTH  = {w_depth_in},
    parameter int WIDTH  = {w_width_in},
    parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic clk_i,
    input  logic [ADDR_W-1:0] addr_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh("{mem_in_w}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[addr_i];
    end

endmodule
"""
        body_b = f"""`timescale 1ns/1ps

module {prefix}_in_bias_rom #(
    parameter int DEPTH  = 1,
    parameter int WIDTH  = {b_width_in},
    parameter int ADDR_W = 1
) (
    input  logic clk_i,
    input  logic [ADDR_W-1:0] addr_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh("{mem_in_b}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[addr_i];
    end

endmodule
"""
        (out_dir / f"{prefix}_in_weights_rom.sv").write_text(body_w, encoding="utf-8")
        (out_dir / f"{prefix}_in_bias_rom.sv").write_text(body_b, encoding="utf-8")
        paths.extend([out_dir / f"{prefix}_in_weights_rom.sv", out_dir / f"{prefix}_in_bias_rom.sv"])

        # fc_out: weights depth=next_in (ROM_DEPTH), bias depth=next_in
        w_depth_out = next_in
        w_width_out = in_f * weight_width
        b_width_out = weight_width * 4
        mem_out_w = f"{mem_path_prefix}{prefix}_out_weights.mem"
        mem_out_b = f"{mem_path_prefix}{prefix}_out_bias.mem"

        body_w_out = f"""`timescale 1ns/1ps

module {prefix}_out_weights_rom #(
    parameter int DEPTH  = {w_depth_out},
    parameter int WIDTH  = {w_width_out},
    parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic clk_i,
    input  logic [ADDR_W-1:0] addr_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh("{mem_out_w}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[addr_i];
    end

endmodule
"""
        body_b_out = f"""`timescale 1ns/1ps

module {prefix}_out_bias_rom #(
    parameter int DEPTH  = {w_depth_out},
    parameter int WIDTH  = {b_width_out},
    parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic clk_i,
    input  logic [ADDR_W-1:0] addr_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh("{mem_out_b}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[addr_i];
    end

endmodule
"""
        (out_dir / f"{prefix}_out_weights_rom.sv").write_text(body_w_out, encoding="utf-8")
        (out_dir / f"{prefix}_out_bias_rom.sv").write_text(body_b_out, encoding="utf-8")
        paths.extend([out_dir / f"{prefix}_out_weights_rom.sv", out_dir / f"{prefix}_out_bias_rom.sv"])

    return paths


# -----------------------------------------------------------------------------
# ROM Module Generation (weight_rom, bias_rom)
# -----------------------------------------------------------------------------

def generate_weight_rom(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    weight_width: int,
    out_dir: Path,
    *,
    mem_subdir: str = "mem_files",
) -> Path:
    """Generate weight_rom_<layer>.sv module in binaryclass_nn format."""
    mem_file = f"{mem_subdir}/{layer_name}_weights.mem"
    if layer_name == "fc1":
        depth = in_features
        packed_width = num_neurons * weight_width
    else:
        depth = num_neurons
        packed_width = in_features * weight_width

    body = f"""`timescale 1ns/1ps

module weight_rom_{layer_name} #(
    parameter int DEPTH  = {depth},
    parameter int WIDTH  = {packed_width},
    parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic clk_i,
    input  logic [ADDR_W-1:0] addr_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh("{mem_file}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[addr_i];
    end

endmodule
"""
    content = _pyramidtech_wrap(body, f"weight_rom_{layer_name}.sv", f"ROM of weights for {layer_name} layer")
    out_path = out_dir / f"weight_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_bias_rom(
    layer_name: str,
    num_neurons: int,
    weight_width: int,
    out_dir: Path,
    *,
    mem_subdir: str = "mem_files",
) -> Path:
    """Generate bias_rom_<layer>.sv module in binaryclass_nn format."""
    mem_file = f"{mem_subdir}/{layer_name}_biases.mem"
    packed_width = num_neurons * weight_width

    body = f"""`timescale 1ns/1ps

module bias_rom_{layer_name} #(
    parameter int WIDTH = {packed_width}
) (
    input  logic clk_i,
    output logic [WIDTH-1:0] data_o
);

    (* rom_style = "block" *)
    logic [WIDTH-1:0] mem [0:0];

    initial begin
        $readmemh("{mem_file}", mem);
    end

    always_ff @(posedge clk_i) begin
        data_o <= mem[0];
    end

endmodule
"""
    content = _pyramidtech_wrap(body, f"bias_rom_{layer_name}.sv", f"ROM of biases for {layer_name} layer")
    out_path = out_dir / f"bias_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# Parameterized fc_in_layer and fc_out_layer (binaryclass_nn style)
# -----------------------------------------------------------------------------

def generate_fc_in_layer_parameterized(
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate single parameterized fc_in_layer.sv with generate blocks for all INPUT_SIZE values."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    if not linear_layers:
        raise RuntimeError("No linear layers for fc_in_layer")
    input_sizes = sorted(set(l.in_features or 0 for l in linear_layers))
    input_size_to_prefix: dict = {}
    for idx, layer in enumerate(linear_layers):
        in_f = layer.in_features or 0
        input_size_to_prefix[in_f] = _proj_prefix(idx)

    weight_blocks = []
    bias_blocks = []
    for inp_sz in input_sizes:
        prefix = input_size_to_prefix[inp_sz]
        cond = "if" if inp_sz == input_sizes[0] else "else if"
        weight_blocks.append(f"""        {cond} (INPUT_SIZE == {inp_sz}) begin
            {prefix}_in_weights_rom #(
                .DEPTH(INPUT_SIZE),
                .WIDTH(NUM_NEURONS*Q_WIDTH)
            ) weights_rom_inst (
                .clk_i(clk),
                .addr_i(weight_addr),
                .data_o(weight_rom_row)
            );
        end""")
        bias_blocks.append(f"""        {cond} (INPUT_SIZE == {inp_sz}) begin
            {prefix}_in_bias_rom #(
                .DEPTH(1),
                .WIDTH(NUM_NEURONS*Q_WIDTH*4)
            ) bias_rom_inst (
                .clk_i(clk),
                .addr_i(1'b0),
                .data_o(bias_rom_row)
            );
        end""")
    first_prefix = input_size_to_prefix[input_sizes[0]]
    weight_blocks.append(f"""        else begin
            {first_prefix}_in_weights_rom #(
                .DEPTH(INPUT_SIZE),
                .WIDTH(NUM_NEURONS*Q_WIDTH)
            ) weights_rom_inst (
                .clk_i(clk),
                .addr_i(weight_addr),
                .data_o(weight_rom_row)
            );
        end""")
    bias_blocks.append(f"""        else begin
            {first_prefix}_in_bias_rom #(
                .DEPTH(1),
                .WIDTH(NUM_NEURONS*Q_WIDTH*4)
            ) bias_rom_inst (
                .clk_i(clk),
                .addr_i(1'b0),
                .data_o(bias_rom_row)
            );
        end""")

    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_in_layer #(
    parameter int NUM_NEURONS = 8,
    parameter int INPUT_SIZE  = 16,
    parameter signed LAYER_SCALE = 12,
    parameter signed BIAS_SCALE  = 0
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,

    output q_data_t data_out[NUM_NEURONS],
    output logic valid_out
);

    logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weight_rom_row;
    logic [NUM_NEURONS*Q_WIDTH*4-1:0] bias_rom_row;
    q_data_t weights_rom_data[NUM_NEURONS];
    acc_t bias_rom_data[NUM_NEURONS];
    logic valid_in_reg;
    q_data_t data_in_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_in_reg <= 1'b0;
            data_in_reg   <= '0;
        end else begin
            valid_in_reg <= valid_in;
            data_in_reg   <= data_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_addr <= '0;
        else if (valid_in)
            weight_addr <= (weight_addr == INPUT_SIZE-1) ? '0 : weight_addr + 1'b1;
    end

    genvar n;
    generate
{chr(10).join(weight_blocks)}
    endgenerate

    generate
{chr(10).join(bias_blocks)}
    endgenerate

    generate
        for (n = 0; n < NUM_NEURONS; n = n + 1) begin : SPLIT_ROM
            assign weights_rom_data[n] = weight_rom_row[n*Q_WIDTH +: Q_WIDTH];
            assign bias_rom_data[n]    = bias_rom_row[n*Q_WIDTH*4 +: Q_WIDTH*4];
        end
    endgenerate

    fc_in #(
        .NUM_NEURONS(NUM_NEURONS),
        .INPUT_SIZE(INPUT_SIZE),
        .LAYER_SCALE(LAYER_SCALE),
        .BIAS_SCALE(BIAS_SCALE)
    ) u_fc_in (
        .clk_i(clk),
        .rst_n_i(rst_n),
        .valid_i(valid_in_reg),
        .data_i(data_in_reg),
        .weights_i(weights_rom_data),
        .biases_i(bias_rom_data),
        .data_o(data_out),
        .valid_o(valid_out)
    );

endmodule
"""
    out_path = out_dir / "fc_in_layer.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


def generate_fc_out_layer_parameterized(
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate single parameterized fc_out_layer.sv with generate blocks for all ROM_DEPTH values."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    if not linear_layers:
        raise RuntimeError("No linear layers for fc_out_layer")
    rom_depths_ordered = []
    rom_depth_to_prefix: dict = {}
    for idx, layer in enumerate(linear_layers):
        next_in = linear_layers[idx + 1].in_features or 1 if idx + 1 < len(linear_layers) else 1
        if next_in not in rom_depth_to_prefix:
            rom_depths_ordered.append(next_in)
        rom_depth_to_prefix[next_in] = _proj_prefix(idx)
    rom_depths = rom_depths_ordered

    weight_blocks = []
    bias_blocks = []
    for rd in rom_depths:
        prefix = rom_depth_to_prefix[rd]
        cond = "if" if rd == rom_depths[0] else "else if"
        addr_arg = "weights_rom_addr" if rd > 1 else "1'b0"
        bias_addr = "bias_rom_addr" if rd > 1 else "1'b0"
        weight_blocks.append(f"""        {cond} (ROM_DEPTH == {rd}) begin
            {prefix}_out_weights_rom #(
                .DEPTH(ROM_DEPTH),
                .WIDTH(NUM_NEURONS*Q_WIDTH)
            ) weights_rom_inst (
                .clk_i(clk),
                .addr_i({addr_arg}),
                .data_o(weights_rom_data_raw)
            );
        end""")
        bias_blocks.append(f"""        {cond} (ROM_DEPTH == {rd}) begin
            {prefix}_out_bias_rom #(
                .DEPTH(ROM_DEPTH),
                .WIDTH(Q_WIDTH*4)
            ) bias_rom_inst (
                .clk_i(clk),
                .addr_i({bias_addr}),
                .data_o(bias_rom_data)
            );
        end""")
    last_prefix = _proj_prefix(len(linear_layers) - 1)
    weight_blocks.append(f"""        else begin
            {last_prefix}_out_weights_rom #(
                .DEPTH(ROM_DEPTH),
                .WIDTH(NUM_NEURONS*Q_WIDTH)
            ) weights_rom_inst (
                .clk_i(clk),
                .addr_i(1'b0),
                .data_o(weights_rom_data_raw)
            );
        end""")
    bias_blocks.append(f"""        else begin
            {last_prefix}_out_bias_rom #(
                .DEPTH(ROM_DEPTH),
                .WIDTH(Q_WIDTH*4)
            ) bias_rom_inst (
                .clk_i(clk),
                .addr_i(1'b0),
                .data_o(bias_rom_data)
            );
        end""")

    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_out_layer #(
    parameter int NUM_NEURONS = 2,
    parameter int ROM_DEPTH   = 45,
    parameter signed LAYER_SCALE = 5,
    parameter signed BIAS_SCALE  = 1
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in[NUM_NEURONS],

    output q_data_t data_out,
    output logic valid_out
);

    localparam int ADDR_W = (ROM_DEPTH > 1) ? $clog2(ROM_DEPTH) : 1;
    logic [ADDR_W-1:0] addr_cnt;
    logic addr_last;
    logic valid_pipeline;
    logic valid_pipeline_reg;
    logic valid_out_engine;
    logic valid_out_engine_reg;
    logic [ADDR_W-1:0] weights_rom_addr;
    logic [ADDR_W-1:0] bias_rom_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weights_rom_data_raw;
    q_data_t weights_rom_data[NUM_NEURONS];
    acc_t bias_rom_data;
    q_data_t weights_rom_data_reg[NUM_NEURONS];
    acc_t bias_rom_data_reg;
    q_data_t data_in_reg[NUM_NEURONS];
    integer k;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            addr_cnt <= '0;
        else if (valid_pipeline)
            addr_cnt <= (addr_cnt == ROM_DEPTH-1) ? '0 : addr_cnt + 1'b1;
    end

    assign addr_last = (addr_cnt == ROM_DEPTH-1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipeline <= 1'b0;
        else if (valid_in)
            valid_pipeline <= 1'b1;
        else if (addr_last)
            valid_pipeline <= 1'b0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipeline_reg   <= 1'b0;
            valid_out_engine_reg <= 1'b0;
        end else begin
            valid_pipeline_reg   <= valid_pipeline;
            valid_out_engine_reg <= valid_out_engine;
        end
    end

    assign valid_out = valid_out_engine_reg;
    assign weights_rom_addr = addr_cnt;
    assign bias_rom_addr    = addr_cnt;

    generate
{chr(10).join(weight_blocks)}
    endgenerate

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : SPLIT_WEIGHTS
            assign weights_rom_data[i] = weights_rom_data_raw[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    generate
{chr(10).join(bias_blocks)}
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                weights_rom_data_reg[k] <= '0;
                data_in_reg[k] <= '0;
            end
            bias_rom_data_reg <= '0;
        end else begin
            for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                weights_rom_data_reg[k] <= weights_rom_data[k];
                data_in_reg[k] <= data_in[k];
            end
            bias_rom_data_reg <= bias_rom_data;
        end
    end

    fc_out #(
        .NUM_NEURONS(NUM_NEURONS),
        .LAYER_SCALE(LAYER_SCALE),
        .BIAS_SCALE(BIAS_SCALE)
    ) u_fc_out (
        .clk_i(clk),
        .rst_n_i(rst_n),
        .valid_i(valid_pipeline_reg),
        .data_i(data_in_reg),
        .weights_i(weights_rom_data_reg),
        .bias_i(bias_rom_data_reg),
        .data_o(data_out),
        .valid_o(valid_out_engine)
    );

endmodule
"""
    out_path = out_dir / "fc_out_layer.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# FC Layer Wrapper Generation (fc_layer_fc1, fc_out_layer_fcN) - legacy per-layer
# -----------------------------------------------------------------------------

def generate_fc_layer_wrapper(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    data_width: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate fc_layer_<layer>.sv wrapper module in binaryclass_nn format."""
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_layer_{layer_name} #(
    parameter int NUM_NEURONS   = {num_neurons},
    parameter int INPUT_SIZE    = {in_features},
    parameter signed BIAS_SCALE  = 0,
    parameter signed LAYER_SCALE = 12
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,

    output q_data_t data_out[NUM_NEURONS],
    output logic valid_out
);

    logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weight_row;
    logic [NUM_NEURONS*Q_WIDTH-1:0] bias_row;

    q_data_t weights[NUM_NEURONS];
    q_data_t bias_raw[NUM_NEURONS];
    acc_t    biases[NUM_NEURONS];

    logic valid_in_reg;
    q_data_t data_in_reg;

    genvar i;

    weight_rom_{layer_name} u_weight_rom (
        .clk_i(clk),
        .addr_i(weight_addr),
        .data_o(weight_row)
    );

    bias_rom_{layer_name} u_bias_rom (
        .clk_i(clk),
        .data_o(bias_row)
    );

    generate
        for (i = 0; i < NUM_NEURONS; i++) begin : WEIGHT_SLICE
            assign weights[i]   = weight_row[i*Q_WIDTH +: Q_WIDTH];
            assign bias_raw[i] = bias_row[i*Q_WIDTH +: Q_WIDTH];
            assign biases[i]   = {{{{(4*Q_WIDTH-Q_WIDTH){{bias_raw[i][Q_WIDTH-1]}}}}, bias_raw[i]}};
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_in_reg <= 1'b0;
            data_in_reg  <= '0;
        end else begin
            valid_in_reg <= valid_in;
            data_in_reg  <= data_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_addr <= '0;
        else if (valid_in)
            weight_addr <= (weight_addr == INPUT_SIZE-1) ? '0 : weight_addr + 1'b1;
    end

    fc_in #(
        .NUM_NEURONS(NUM_NEURONS),
        .INPUT_SIZE(INPUT_SIZE),
        .BIAS_SCALE(BIAS_SCALE),
        .LAYER_SCALE(LAYER_SCALE)
    ) u_fc_in (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in_reg),
        .data_in(data_in_reg),
        .weights(weights),
        .biases(biases),
        .data_out(data_out),
        .valid_out(valid_out)
    );

endmodule
"""
    content = _pyramidtech_wrap(body, f"fc_layer_{layer_name}.sv", f"Fully-connected input layer {layer_name} with sequential ROM access.")
    out_path = out_dir / f"fc_layer_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_fc_out_layer(
    layer_name: str,
    num_neurons: int,
    num_inputs: int,
    out_dir: Path
) -> Path:
    """Generate fc_out_layer_<layer>.sv in binaryclass_nn format."""
    data_o_port = "output q_data_t data_out" if num_neurons == 1 else "output q_data_t data_out[NUM_NEURONS]"
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_out_layer_{layer_name} #(
   parameter int NUM_NEURONS   = {num_neurons},
   parameter int NUM_INPUTS    = {num_inputs},
   parameter signed BIAS_SCALE  = 1,
   parameter signed LAYER_SCALE = 5
)(
   input  logic clk,
   input  logic rst_n,
   input  logic valid_in,
   input  q_data_t data_in[NUM_INPUTS],
   {data_o_port},
   output logic valid_out
);

   logic [$clog2(NUM_NEURONS)-1:0] addr_cnt;
   logic valid_in_reg;
   q_data_t data_in_reg[NUM_INPUTS];

   logic [NUM_INPUTS*Q_WIDTH-1:0] weights_rom_data_raw;
   q_data_t weights_rom_data[NUM_INPUTS];
   logic [NUM_NEURONS*Q_WIDTH-1:0] bias_rom_data_raw;
   q_data_t bias_raw[NUM_NEURONS];
   acc_t bias_reg;

   logic valid_pipeline_reg;
   logic [$clog2(NUM_NEURONS)-1:0] addr_cnt_d;
   logic fc_out_valid;
   q_data_t fc_out_tmp;
   logic [$clog2(NUM_NEURONS+1)-1:0] fc_out_cnt;
   q_data_t fc_out_buf[NUM_NEURONS];

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         addr_cnt <= '0;
      else if (addr_cnt == NUM_NEURONS-1)
         addr_cnt <= '0;
      else if (valid_in || addr_cnt > 0)
         addr_cnt <= addr_cnt + 1'b1;
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         valid_in_reg <= 1'b0;
         for (int j = 0; j < NUM_INPUTS; j++)
            data_in_reg[j] <= '0;
      end else begin
         if (addr_cnt == '0) begin
            valid_in_reg <= valid_in;
            data_in_reg  <= data_in;
         end else if (addr_cnt == NUM_NEURONS) begin
            valid_in_reg <= 1'b0;
            for (int j = 0; j < NUM_INPUTS; j++)
               data_in_reg[j] <= '0;
         end
      end
   end

   weight_rom_{layer_name} u_weights_rom (
      .clk_i(clk),
      .addr_i(addr_cnt),
      .data_o(weights_rom_data_raw)
   );

   bias_rom_{layer_name} u_bias_rom (
      .clk_i(clk),
      .data_o(bias_rom_data_raw)
   );

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         valid_pipeline_reg <= 1'b0;
         addr_cnt_d        <= '0;
      end else begin
         valid_pipeline_reg <= valid_in_reg;
         addr_cnt_d         <= addr_cnt;
      end
   end

   genvar i, k;
   generate
      for (i = 0; i < NUM_INPUTS; i++) begin : INPUT_SPLIT
         assign weights_rom_data[i] = weights_rom_data_raw[i*Q_WIDTH +: Q_WIDTH];
      end
      for (k = 0; k < NUM_NEURONS; k++) begin : SPLIT_BIAS
         assign bias_raw[k] = bias_rom_data_raw[k*Q_WIDTH +: Q_WIDTH];
      end
   endgenerate

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         bias_reg <= '0;
      else
         bias_reg <= {{(4*Q_WIDTH-Q_WIDTH){{bias_raw[addr_cnt_d][Q_WIDTH-1]}}}}, bias_raw[addr_cnt_d]}};
   end

   fc_out #(
      .NUM_NEURONS(NUM_INPUTS),
      .LAYER_SCALE(LAYER_SCALE),
      .BIAS_SCALE(BIAS_SCALE)
   ) u_fc_out (
      .clk(clk),
      .rst_n(rst_n),
      .valid_in(valid_pipeline_reg),
      .data_in(data_in_reg),
      .weights(weights_rom_data),
      .bias(bias_reg),
      .data_out(fc_out_tmp),
      .valid_out(fc_out_valid)
   );

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         for (int n = 0; n < NUM_NEURONS; n++)
            fc_out_buf[n] <= '0;
         fc_out_cnt <= '0;
         valid_out  <= 1'b0;
      end else begin
         valid_out <= 1'b0;
         if (fc_out_valid) begin
            for (int n = 0; n < NUM_NEURONS-1; n++)
               fc_out_buf[n] <= fc_out_buf[n+1];
            fc_out_buf[NUM_NEURONS-1] <= fc_out_tmp;
            if (fc_out_cnt == NUM_NEURONS-1) begin
               valid_out   <= 1'b1;
               fc_out_cnt  <= '0;
            end else if (fc_out_cnt < NUM_NEURONS)
               fc_out_cnt <= fc_out_cnt + 1'b1;
         end
      end
   end

   assign data_out = (NUM_NEURONS == 1) ? fc_out_buf[0] : fc_out_buf;

endmodule
"""
    content = _pyramidtech_wrap(body, f"fc_out_layer_{layer_name}.sv", f"Fully-connected output layer {layer_name} with sequential ROM access.")
    out_path = out_dir / f"fc_out_layer_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# Top Module Generation (hierarchical and flattened)
# -----------------------------------------------------------------------------

def _generate_top_module_binaryclass_nn_style(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    out_dir: Path,
    linear_layers: List[LayerInfo],
    *,
    final_activation: Optional[str] = None,
    python_scale: int = 256,
) -> Path:
    """Generate top module in binaryclass_nn style: fc_in -> fc_out -> relu (scalar) per layer.
    Final activation: Sigmoid (default for binary), Relu, or pass-through if unsupported.
    LAYER_SCALE computed from dimensions and ONNX quant params."""
    port_decls = [
        "    input  logic clk,",
        "    input  logic rst_n,",
        "    input  logic valid_in,",
        "    input  q_data_t data_in,",
        "    output logic valid_out,",
        "    output q_data_t data_out",
    ]
    signal_decls: List[str] = []
    instantiations: List[str] = []
    connections: List[str] = []

    stream_valid = "valid_in"
    stream_data = "data_in"

    for i, fc in enumerate(linear_layers):
        fc_name = fc.name
        num_neurons = fc.out_features or 0
        in_features = fc.in_features or 0
        next_in = linear_layers[i + 1].in_features or 1 if i + 1 < len(linear_layers) else 1

        fc_out_vec = f"{fc_name}_out"
        fc_valid = f"{fc_name}_valid_out"
        pre_act = f"{fc_name}_pre_act"
        post_act = f"{fc_name}_post_act"
        act_valid_i = f"{fc_name}_act_valid_i"
        act_valid_o = f"{fc_name}_act_valid_o"

        signal_decls.extend([
            f"    logic        {fc_valid};",
            _q_data_decl(fc_out_vec, num_neurons, "    "),
            f"    q_data_t     {pre_act};",
            f"    logic        {act_valid_i};",
        ])
        if i < len(linear_layers) - 1:
            signal_decls.extend([
                f"    q_data_t     {post_act};",
                f"    logic        {act_valid_o};",
                "",
            ])
        else:
            signal_decls.append("")
        signal_decls.append("")

        # fc_in_layer: LAYER_SCALE from input_size (MAC accumulator scaling)
        layer_scale_in = compute_layer_scale(
            weight_width, in_features,
            fc.quant_params.get("weight_scale") if fc.quant_params else None,
            python_scale,
        )
        # fc_out_layer: LAYER_SCALE from ROM_DEPTH (next_in)
        layer_scale_out = compute_layer_scale(weight_width, next_in, None, python_scale)
        bias_scale_out = compute_bias_scale(weight_width, next_in, python_scale)

        # fc_in_layer
        instantiations.extend([
            f"    fc_in_layer #(",
            f"        .NUM_NEURONS({num_neurons}),",
            f"        .INPUT_SIZE({in_features}),",
            f"        .BIAS_SCALE(0),",
            f"        .LAYER_SCALE({layer_scale_in})",
            f"    ) u_{fc_name}_in (",
            f"        .clk(clk),",
            f"        .rst_n(rst_n),",
            f"        .valid_in({stream_valid}),",
            f"        .data_in({stream_data}),",
            f"        .data_out({fc_out_vec}),",
            f"        .valid_out({fc_valid})",
            f"    );",
            "",
        ])

        # fc_out_layer
        instantiations.extend([
            f"    fc_out_layer #(",
            f"        .NUM_NEURONS({num_neurons}),",
            f"        .ROM_DEPTH({next_in}),",
            f"        .LAYER_SCALE({layer_scale_out}),",
            f"        .BIAS_SCALE({bias_scale_out})",
            f"    ) u_{fc_name}_out (",
            f"        .clk(clk),",
            f"        .rst_n(rst_n),",
            f"        .valid_in({fc_valid}),",
            f"        .data_in({fc_out_vec}),",
            f"        .data_out({pre_act}),",
            f"        .valid_out({act_valid_i})",
            f"    );",
            "",
        ])

        # Per-layer activation from ONNX (or default)
        layer_act = fc.activation if fc.activation else ("Relu" if i < len(linear_layers) - 1 else final_activation or "Sigmoid")
        if i < len(linear_layers) - 1:
            # Intermediate: use detected activation or default Relu
            if layer_act == "Relu":
                instantiations.extend([
                    f"    relu_layer u_relu_{i+1} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({act_valid_i}),",
                    f"        .data_in({pre_act}),",
                    f"        .data_out({post_act}),",
                    f"        .valid_out({act_valid_o})",
                    f"    );",
                    "",
                ])
            elif layer_act in ("Tanh", "HardSigmoid"):
                LOGGER.warning(f"Activation '{layer_act}' for {fc.name} not implemented in RTL; using Relu.")
                layer_act = "Relu"
                instantiations.extend([
                    f"    relu_layer u_relu_{i+1} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({act_valid_i}),",
                    f"        .data_in({pre_act}),",
                    f"        .data_out({post_act}),",
                    f"        .valid_out({act_valid_o})",
                    f"    );",
                    "",
                ])
            else:
                instantiations.extend([
                    f"    relu_layer u_relu_{i+1} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({act_valid_i}),",
                    f"        .data_in({pre_act}),",
                    f"        .data_out({post_act}),",
                    f"        .valid_out({act_valid_o})",
                    f"    );",
                    "",
                ])
            stream_valid = act_valid_o
            stream_data = post_act
        else:
            # Final activation (from ONNX per-layer or final_activation or default Sigmoid)
            use_sigmoid = layer_act in (None, "Sigmoid")
            use_relu = layer_act == "Relu"
            if layer_act not in (None, "Sigmoid", "Relu") and layer_act:
                LOGGER.warning(
                    f"Final activation '{layer_act}' from ONNX not supported in RTL; "
                    "using Sigmoid (binary classifier default)."
                )
                use_sigmoid = True
            if use_sigmoid:
                instantiations.extend([
                    f"    sigmoid_layer u_sigmoid (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({act_valid_i}),",
                    f"        .data_in({pre_act}),",
                    f"        .data_out(data_out),",
                    f"        .valid_out(valid_out)",
                    f"    );",
                    "",
                ])
            elif use_relu:
                instantiations.extend([
                    f"    relu_layer u_relu_final (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({act_valid_i}),",
                    f"        .data_in({pre_act}),",
                    f"        .data_out(data_out),",
                    f"        .valid_out(valid_out)",
                    f"    );",
                    "",
                ])
            else:
                instantiations.extend([
                    f"    assign data_out = {pre_act};",
                    f"    assign valid_out = {act_valid_i};",
                    "",
                ])

    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module {model_name}_top (
{chr(10).join(port_decls)}
);

{chr(10).join(signal_decls)}

{chr(10).join(connections)}

{chr(10).join(instantiations)}

endmodule
"""
    content = _pyramidtech_wrap(body, f"{model_name}_top.sv", f"Top-level binary classification NN (binaryclass_nn style) for {model_name}.")
    out_path = out_dir / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_top_module(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    out_dir: Path,
    *,
    has_softmax: bool = False,
    use_parameterized_layers: bool = True,
    use_binaryclass_nn_style: bool = False,
    final_activation: Optional[str] = None,
    python_scale: int = 256,
) -> Path:
    """Generate top module with streaming control logic. has_softmax is accepted but ignored.
    use_binaryclass_nn_style: fc_in -> fc_out -> relu (scalar) per layer, sigmoid at end.
    final_activation: from ONNX graph (Sigmoid, Relu, etc.) or None for default.
    python_scale: used to compute LAYER_SCALE from dimensions."""
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    linear_layers = [l for l in active_layers if l.layer_type == "linear"]

    if use_binaryclass_nn_style and use_parameterized_layers:
        return _generate_top_module_binaryclass_nn_style(
            model_name, layers, input_size, data_width, weight_width, out_dir, linear_layers,
            final_activation=final_activation,
            python_scale=python_scale,
        )

    port_decls = [
        "    input  logic clk,",
        "    input  logic rst_n,",
        "    input  logic valid_in,",
        "    input  q_data_t data_in,",
    ]

    signal_decls = []
    instantiations = []
    connections = []

    stream_valid = "valid_in"
    stream_data = "data_in"
    stream_vector = None
    stream_vector_valid = None
    prev_fc_out = None
    prev_fc_valid = None

    for i, layer in enumerate(active_layers):
        if layer.layer_type == "linear":
            fc_name = layer.name
            num_neurons = layer.out_features
            in_features = layer.in_features

            fc_valid_out = f"{fc_name}_valid_out"
            fc_out = f"{fc_name}_out"

            signal_decls.extend([
                f"    logic        {fc_valid_out};",
                _q_data_decl(fc_out, num_neurons, "    "),
                ""
            ])

            if use_parameterized_layers:
                if fc_name == "fc1":
                    layer_scale_fc1 = compute_layer_scale(
                        weight_width, in_features,
                        layer.quant_params.get("weight_scale") if layer.quant_params else None,
                        256,
                    )
                    instantiations.extend([
                        f"    fc_in_layer #(",
                        f"        .NUM_NEURONS({num_neurons}),",
                        f"        .INPUT_SIZE({in_features}),",
                        f"        .BIAS_SCALE(0),",
                        f"        .LAYER_SCALE({layer_scale_fc1})",
                        f"    ) u_{fc_name}_in (",
                        f"        .clk(clk),",
                        f"        .rst_n(rst_n),",
                        f"        .valid_in({stream_valid}),",
                        f"        .data_in({stream_data}),",
                        f"        .data_out({fc_out}),",
                        f"        .valid_out({fc_valid_out})",
                        f"    );",
                        ""
                    ])
                else:
                    layer_scale_out = compute_layer_scale(weight_width, num_neurons, None, 256)
                    bias_scale_out = compute_bias_scale(weight_width, num_neurons, 256)
                    connections.append(f"    // fc_out_layer for {fc_name}")
                    connections.append("")
                    instantiations.extend([
                        f"    fc_out_layer #(",
                        f"        .NUM_NEURONS({in_features}),",
                        f"        .ROM_DEPTH({num_neurons}),",
                        f"        .LAYER_SCALE({layer_scale_out}),",
                        f"        .BIAS_SCALE({bias_scale_out})",
                        f"    ) u_{fc_name}_out (",
                        f"        .clk(clk),",
                        f"        .rst_n(rst_n),",
                        f"        .valid_in({stream_vector_valid}),",
                        f"        .data_in({stream_vector}),",
                        f"        .data_out({fc_out}),",
                        f"        .valid_out({fc_valid_out})",
                        f"    );",
                        ""
                    ])
            elif fc_name == "fc1":
                instantiations.extend([
                    f"    fc_layer_{fc_name} #(",
                    f"        .NUM_NEURONS  ({num_neurons}),",
                    f"        .INPUT_SIZE   ({in_features})",
                    f"    ) u_{fc_name} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({stream_valid}),",
                    f"        .data_in({stream_data}),",
                    f"        .data_out({fc_out}),",
                    f"        .valid_out({fc_valid_out})",
                    f"    );",
                    ""
                ])
            else:
                connections.append(f"    // fc_out_layer_{fc_name} uses {stream_vector} and {stream_vector_valid}")
                connections.append("")
                instantiations.extend([
                    f"    fc_out_layer_{fc_name} #(",
                    f"        .NUM_NEURONS ({num_neurons}),",
                    f"        .NUM_INPUTS  ({in_features})",
                    f"    ) u_{fc_name} (",
                    f"        .clk(clk),",
                    f"        .rst_n(rst_n),",
                    f"        .valid_in({stream_vector_valid}),",
                    f"        .data_in({stream_vector}),",
                    f"        .data_out({fc_out}),",
                    f"        .valid_out({fc_valid_out})",
                    f"    );",
                    ""
                ])

            has_relu = (i + 1 < len(active_layers) and
                       active_layers[i + 1].layer_type == "relu")

            if has_relu:
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
            else:
                prev_fc_out = fc_out
                prev_fc_valid = fc_valid_out
                break

        elif layer.layer_type == "relu":
            if prev_fc_out is None or prev_fc_valid is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer output")

            prev_fc = None
            for j in range(i-1, -1, -1):
                if active_layers[j].layer_type == "linear":
                    prev_fc = active_layers[j]
                    break

            if prev_fc is None:
                raise RuntimeError(f"ReLU {layer.name} has no preceding FC layer")

            num_neurons = prev_fc.out_features
            relu_name = layer.name
            relu_in_vector = prev_fc_out
            relu_in_valid = prev_fc_valid
            relu_out_vector = f"{relu_name}_out"
            relu_out_valid = f"{relu_name}_valid_out"

            signal_decls.extend([
                _q_data_decl(relu_out_vector, num_neurons, "    "),
                f"    logic      {relu_out_valid};",
                ""
            ])

            instantiations.extend([
                f"    relu_layer_array #(",
                f"        .NUM_NEURONS({num_neurons})",
                f"    ) u_{relu_name} (",
                f"        .clk(clk),",
                f"        .rst_n(rst_n),",
                f"        .valid_in({relu_in_valid}),",
                f"        .data_in({relu_in_vector}),",
                f"        .data_out({relu_out_vector}),",
                f"        .valid_out({relu_out_valid})",
                f"    );",
                ""
            ])

            stream_vector = relu_out_vector
            stream_vector_valid = relu_out_valid

    last_fc = None
    for layer in reversed(active_layers):
        if layer.layer_type == "linear":
            last_fc = layer
            break

    if last_fc is None:
        raise RuntimeError("No output FC layer found")

    output_size = last_fc.out_features
    last_fc_out = f"{last_fc.name}_out"
    last_fc_valid = f"{last_fc.name}_valid_out"

    data_out_line = _q_data_port_decl("data_out", output_size, "output").replace("    ", "    ")
    port_decls.extend([
        "    output logic valid_out,",
        data_out_line
    ])
    connections.extend([
        f"    assign valid_out = {last_fc_valid};",
        f"    assign data_out  = {last_fc_out};"
    ])

    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module {model_name}_top (
{chr(10).join(port_decls)}
);

{chr(10).join(signal_decls)}

{chr(10).join(connections)}

{chr(10).join(instantiations)}

endmodule
"""
    content = _pyramidtech_wrap(body, f"{model_name}_top.sv", f"Top-level binary classification neural network for {model_name}.")

    out_path = out_dir / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_flattened_top_module(
    model_name: str,
    layers: List[LayerInfo],
    input_size: int,
    data_width: int,
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate a single flattened top module with all logic inlined (no submodule instances)."""
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    linear_layers = [l for l in active_layers if l.layer_type == "linear"]
    if len(linear_layers) < 3:
        raise RuntimeError(
            f"Flattened generation requires at least 3 FC layers; found {len(linear_layers)}"
        )
    for i, fc in enumerate(linear_layers):
        if fc.name != f"fc{i + 1}":
            raise RuntimeError(f"Flattened expects fc1..fcN; got {fc.name}")

    num_layers = len(linear_layers)
    n_list = [fc.out_features or 0 for fc in linear_layers]
    in_list = [linear_layers[0].in_features or input_size] + [linear_layers[i].out_features or 0 for i in range(num_layers - 1)]

    mem_prefix = "mem_files"
    output_size = n_list[-1]

    parts: List[str] = []
    localparam_lines = []
    for i in range(num_layers):
        localparam_lines.append(f"    localparam FC{i+1}_NEURONS    = {n_list[i]};")
    for i in range(num_layers):
        localparam_lines.append(f"    localparam FC{i+1}_INPUT_SIZE = {in_list[i]}")

    out_data_port = _q_data_port_decl("data_out", output_size, "output")
    parts.append(f'''`timescale 1ns/1ps
import quant_pkg::*;

module {{model_name}}_top (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,
    output logic valid_out,
    {out_data_port}
);

{chr(10).join(localparam_lines)}
''')

    parts.append(f'''
    logic [$clog2(FC1_INPUT_SIZE)-1:0] fc1_weight_addr;
    logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_weight_row;
    logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_bias_row;
    q_data_t fc1_weights [FC1_NEURONS];
    q_data_t fc1_bias    [FC1_NEURONS];
    acc_t fc1_mac_acc_tmp [FC1_NEURONS];
    acc_t fc1_acc_tmp [FC1_NEURONS];
    acc_t fc1_bias_32 [FC1_NEURONS];
    logic [$clog2(FC1_INPUT_SIZE+1)-1:0] fc1_out_count;
    logic [FC1_NEURONS-1:0] fc1_mac_out_valid;
    logic fc1_all_mac_valid;
    logic signed [63:0] fc1_mult_reg [FC1_NEURONS];
    logic signed [63:0] fc1_acc_reg [FC1_NEURONS];
    logic fc1_enable_q, fc1_enable_qq;
    q_data_t fc1_out [FC1_NEURONS];
    logic fc1_valid_out;

    (* rom_style = "block" *) logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_mem_weight [0:FC1_INPUT_SIZE-1];
    (* rom_style = "block" *) logic [FC1_NEURONS*Q_WIDTH-1:0] fc1_mem_bias [0:0];

    initial begin
        $readmemh("{mem_prefix}/fc1_weights.mem", fc1_mem_weight);
        $readmemh("{mem_prefix}/fc1_biases.mem", fc1_mem_bias);
    end

    genvar gi;
    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_WEIGHT_SLICE
            assign fc1_weights[gi] = fc1_weight_row[gi*Q_WIDTH +: Q_WIDTH];
            assign fc1_bias[gi] = fc1_bias_row[gi*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate
    assign fc1_weight_row = fc1_mem_weight[fc1_weight_addr];
    assign fc1_bias_row = fc1_mem_bias[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc1_weight_addr <= '0;
        else if (valid_in)
            fc1_weight_addr <= (fc1_weight_addr == FC1_INPUT_SIZE-1) ? '0 : fc1_weight_addr + 1'b1;
    end

    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_MAC
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_mult_reg[gi] <= '0;
                else if (valid_in) fc1_mult_reg[gi] <= $signed(data_in) * $signed(fc1_weights[gi]);
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_acc_reg[gi] <= '0;
                else if (fc1_valid_out) fc1_acc_reg[gi] <= '0;
                else if (valid_in) fc1_acc_reg[gi] <= fc1_acc_reg[gi] + fc1_mult_reg[gi];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else if (fc1_valid_out) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else begin fc1_enable_q <= valid_in; fc1_enable_qq <= fc1_enable_q; end
    end

    generate for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_MAC_VALID
        assign fc1_mac_out_valid[gi] = fc1_enable_qq;
    end endgenerate
    assign fc1_all_mac_valid = &fc1_mac_out_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc1_out_count <= '0; fc1_valid_out <= 1'b0; end
        else begin
            fc1_valid_out <= 1'b0;
            if (fc1_all_mac_valid) begin
                if (fc1_out_count == FC1_INPUT_SIZE-1) begin
                    fc1_out_count <= '0;
                    fc1_valid_out <= 1'b1;
                end else fc1_out_count <= fc1_out_count + 1'b1;
            end
        end
    end

    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_BIAS_ADD
            assign fc1_mac_acc_tmp[gi] = ((fc1_acc_reg[gi] > $signed(32'h7FFFFFFF)) ? 32'sh7FFFFFFF :
                ((fc1_acc_reg[gi] < $signed(32'h80000000)) ? 32'sh80000000 : fc1_acc_reg[gi][31:0])) >>> FRAC_WIDTH;
            assign fc1_bias_32[gi] = {{ {{64-Q_WIDTH{{fc1_bias[gi][Q_WIDTH-1]}}}}, fc1_bias[gi] }};
            assign fc1_acc_tmp[gi] = fc1_mac_acc_tmp[gi] + fc1_bias_32[gi];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_out[gi] <= '0;
                else if (fc1_out_count == FC1_INPUT_SIZE-1) begin
                    if (fc1_acc_tmp[gi] > SATURATION_H) fc1_out[gi] <= LIMIT_H;
                    else if (fc1_acc_tmp[gi] < SATURATION_L) fc1_out[gi] <= $signed(LIMIT_L);
                    else fc1_out[gi] <= fc1_acc_tmp[gi][Q_WIDTH:0];
                end
            end
        end
    endgenerate
''')

    parts.append(f'''
    q_data_t relu1_out [FC1_NEURONS];
    logic relu1_valid_out;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) relu1_valid_out <= 1'b0;
        else relu1_valid_out <= fc1_valid_out;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (int i=0; i<FC1_NEURONS; i++) relu1_out[i] <= '0;
        else if (fc1_valid_out)
            for (int i=0; i<FC1_NEURONS; i++)
                relu1_out[i] <= (fc1_out[i] < 0) ? '0 : fc1_out[i];
    end
''')

    for idx in range(1, num_layers):
        i = idx + 1
        n_neurons = n_list[idx]
        n_inputs = in_list[idx]
        prev_relu = f"relu{idx}_out" if idx > 1 else "relu1_out"
        prev_valid = f"relu{idx}_valid_out" if idx > 1 else "relu1_valid_out"
        is_last = idx == num_layers - 1
        sum_comb_line = ("        fc{i}_sum_comb = {{ {{64-Q_WIDTH{{".format(i=i) +
                        f"fc{i}_bias_reg" + "[Q_WIDTH-1]" + "}}}}, " + f"fc{i}_bias_reg" + " }};")

        fc_block = f'''
    logic [$clog2(FC{i}_NEURONS)-1:0] fc{i}_addr_cnt;
    logic fc{i}_valid_in_reg;
    q_data_t fc{i}_fc_in_reg [FC{i}_INPUT_SIZE];
    logic [FC{i}_INPUT_SIZE*Q_WIDTH-1:0] fc{i}_weights_raw;
    q_data_t fc{i}_weights [FC{i}_INPUT_SIZE];
    logic [FC{i}_NEURONS*Q_WIDTH-1:0] fc{i}_bias_raw;
    q_data_t fc{i}_bias [FC{i}_NEURONS];
    q_data_t fc{i}_bias_reg;
    logic signed [63:0] fc{i}_mult_res [FC{i}_INPUT_SIZE];
    logic signed [63:0] fc{i}_mult_tmp [FC{i}_INPUT_SIZE];
    logic signed [63:0] fc{i}_sum_comb;
    logic fc{i}_out_valid;
    q_data_t fc{i}_out_tmp;
    q_data_t fc{i}_out [FC{i}_NEURONS];
    logic [$clog2(FC{i}_NEURONS+1)-1:0] fc{i}_out_cnt;
    logic fc{i}_valid_out;

    (* rom_style = "block" *) logic [FC{i}_INPUT_SIZE*Q_WIDTH-1:0] fc{i}_mem_weight [0:FC{i}_NEURONS-1];
    (* rom_style = "block" *) logic [FC{i}_NEURONS*Q_WIDTH-1:0] fc{i}_mem_bias [0:0];

    initial begin
        $readmemh("{mem_prefix}/fc{i}_weights.mem", fc{i}_mem_weight);
        $readmemh("{mem_prefix}/fc{i}_biases.mem", fc{i}_mem_bias);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc{i}_addr_cnt <= '0;
        else if (fc{i}_addr_cnt == FC{i}_NEURONS-1) fc{i}_addr_cnt <= '0;
        else if ({prev_valid} || fc{i}_addr_cnt > 0) fc{i}_addr_cnt <= fc{i}_addr_cnt + 1'b1;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc{i}_valid_in_reg <= 1'b0;
            for (int j=0; j<FC{i}_INPUT_SIZE; j++) fc{i}_fc_in_reg[j] <= '0;
        end else begin
            if (fc{i}_addr_cnt == '0) begin
                fc{i}_valid_in_reg <= {prev_valid};
                fc{i}_fc_in_reg <= {prev_relu};
            end else if (fc{i}_addr_cnt == FC{i}_NEURONS) begin
                fc{i}_valid_in_reg <= 1'b0;
                for (int j=0; j<FC{i}_INPUT_SIZE; j++) fc{i}_fc_in_reg[j] <= '0;
            end
        end
    end

    assign fc{i}_weights_raw = fc{i}_mem_weight[fc{i}_addr_cnt];
    assign fc{i}_bias_raw = fc{i}_mem_bias[0];
    generate for (gi=0; gi<FC{i}_INPUT_SIZE; gi++) assign fc{i}_weights[gi] = fc{i}_weights_raw[gi*Q_WIDTH +: Q_WIDTH]; endgenerate
    generate for (gi=0; gi<FC{i}_NEURONS; gi++) assign fc{i}_bias[gi] = fc{i}_bias_raw[gi*Q_WIDTH +: Q_WIDTH]; endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc{i}_bias_reg <= '0;
        else fc{i}_bias_reg <= fc{i}_bias[fc{i}_addr_cnt];
    end

    generate for (gi=0; gi<FC{i}_INPUT_SIZE; gi++) begin : FC{i}_MULT
        assign fc{i}_mult_res[gi] = $signed(fc{i}_fc_in_reg[gi]) * $signed(fc{i}_weights[gi]);
        assign fc{i}_mult_tmp[gi] = (fc{i}_mult_res[gi] >>> FRAC_WIDTH);
    end endgenerate

    integer fc{i}_k;
    always_comb begin
        {sum_comb_line}
        for (fc{i}_k=0; fc{i}_k<FC{i}_INPUT_SIZE; fc{i}_k++) fc{i}_sum_comb = fc{i}_sum_comb + fc{i}_mult_tmp[fc{i}_k];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc{i}_out_tmp <= '0; fc{i}_out_valid <= 1'b0; end
        else if (fc{i}_valid_in_reg) begin
            fc{i}_out_valid <= 1'b1;
            if (fc{i}_sum_comb > SATURATION_H) fc{i}_out_tmp <= LIMIT_H;
            else if (fc{i}_sum_comb < SATURATION_L) fc{i}_out_tmp <= $signed(LIMIT_L);
            else fc{i}_out_tmp <= fc{i}_sum_comb[Q_WIDTH:0];
        end else fc{i}_out_valid <= 1'b0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int n=0; n<FC{i}_NEURONS; n++) fc{i}_out[n] <= '0;
            fc{i}_out_cnt <= '0; fc{i}_valid_out <= 1'b0;
        end else begin
            fc{i}_valid_out <= 1'b0;
            if (fc{i}_out_valid) begin
                for (int n=0; n<FC{i}_NEURONS-1; n++) fc{i}_out[n] <= fc{i}_out[n+1];
                fc{i}_out[FC{i}_NEURONS-1] <= fc{i}_out_tmp;
                if (fc{i}_out_cnt == FC{i}_NEURONS-1) begin fc{i}_valid_out <= 1'b1; fc{i}_out_cnt <= '0; end
                else if (fc{i}_out_cnt < FC{i}_NEURONS) fc{i}_out_cnt <= fc{i}_out_cnt + 1'b1;
            end
        end
    end
'''
        parts.append(fc_block)

        if is_last:
            parts.append(f'''
    assign valid_out = fc{i}_valid_out;
    assign data_out = fc{i}_out;
''')
        else:
            parts.append(f'''
    q_data_t relu{i}_out [FC{i}_NEURONS];
    logic relu{i}_valid_out;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) relu{i}_valid_out <= 1'b0;
        else relu{i}_valid_out <= fc{i}_valid_out;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) for (int j=0; j<FC{i}_NEURONS; j++) relu{i}_out[j] <= '0;
        else if (fc{i}_valid_out)
            for (int j=0; j<FC{i}_NEURONS; j++)
                relu{i}_out[j] <= (fc{i}_out[j] < 0) ? '0 : fc{i}_out[j];
    end
''')

    parts.append("endmodule\n")
    content = "\n".join(parts).replace("{model_name}", model_name)
    out_path = out_dir / f"{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_axi4_stream_wrapper(
    model_name: str,
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate AXI4-Stream wrapper for hierarchical top (binaryclass_NN-style interface).
    Wraps {model_name}_top with s_axis_* / m_axis_prediction_* ports."""
    last_fc = next((l for l in reversed(layers) if l.layer_type == "linear"), None)
    if not last_fc:
        raise RuntimeError("No linear layer for AXI4 wrapper")
    output_size = last_fc.out_features or 0

    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module {model_name}_axi4_wrapper (
    input  logic        clk_i,
    input  logic        rst_n_i,

    // AXI4-Stream Slave Interface: Data applied for prediction
    input  q_data_t     s_axis_tdata_i,
    input  logic        s_axis_tvalid_i,
    input  logic        s_axis_tlast_i,

    // AXI4-Stream Master Interface: Prediction result data
    output logic [31:0] m_axis_prediction_tdata_o,
    output logic [3:0]  m_axis_prediction_tkeep_o,
    output logic        m_axis_prediction_tvalid_o,
    input  logic        m_axis_prediction_tready_i,
    output logic        m_axis_prediction_tlast_o
);

    q_data_t prediction;
    logic    prediction_valid;

    {model_name}_top u_top (
        .clk(clk_i),
        .rst_n(rst_n_i),
        .valid_in(s_axis_tvalid_i),
        .data_in(s_axis_tdata_i),
        .valid_out(prediction_valid),
        .data_out(prediction)
    );

    assign m_axis_prediction_tdata_o = {{{{(32-Q_WIDTH){{1'b0}}}}, prediction}};
    assign m_axis_prediction_tkeep_o = 4'h1;
    assign m_axis_prediction_tvalid_o = prediction_valid;
    assign m_axis_prediction_tlast_o  = prediction_valid;

endmodule
"""
    content = _pyramidtech_wrap(
        body,
        f"{model_name}_axi4_wrapper.sv",
        f"AXI4-Stream wrapper for {model_name} (binaryclass_NN-style interface)",
    )
    out_path = out_dir / f"{model_name}_axi4_wrapper.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_wrapper_module(
    model_name: str,
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate a thin wrapper module that instantiates the top module (used in flattened flow)."""
    last_fc = next((l for l in reversed(layers) if l.layer_type == "linear"), None)
    if not last_fc:
        raise RuntimeError("No linear layer for wrapper")
    output_size = last_fc.out_features or 0

    out_data_port = _q_data_port_decl("out_data", output_size, "output")
    content = f'''`timescale 1ns/1ps
import quant_pkg::*;

module {{model_name}}_wrapper (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  q_data_t     in_data,
    output logic        out_valid,
    {out_data_port}
);

    {{model_name}}_top u_top (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(in_valid),
        .data_in(in_data),
        .valid_out(out_valid),
        .data_out(out_data)
    );

endmodule
'''
    content = content.replace("{model_name}", model_name)
    out_path = out_dir / f"{model_name}_wrapper.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# Testbench and Report Generation
# -----------------------------------------------------------------------------

def generate_testbench(
    model_name: str,
    input_size: int,
    output_size: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate simple testbench."""
    out_data_decl = _q_data_decl("out_data", output_size).replace(
        f" [{output_size}]", " [OUTPUT_SIZE]"
    ) if output_size > 1 else _q_data_decl("out_data", output_size)
    content = f"""`timescale 1ns/1ps
`define Q_INT{weight_width}
import quant_pkg::*;

module tb_{model_name}_top;

    parameter int INPUT_SIZE = {input_size};
    parameter int OUTPUT_SIZE = {output_size};

    logic clk;
    logic rst_n;
    logic in_valid;
    q_data_t in_data;
    logic out_valid;
    {out_data_decl}

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        #100;
        rst_n = 1;
    end

    {model_name}_top u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(in_valid),
        .data_in(in_data),
        .valid_out(out_valid),
        .data_out(out_data)
    );

    initial begin
        in_valid = 0;
        in_data = 0;

        wait(rst_n);
        #20;

        for (int i = 0; i < INPUT_SIZE; i++) begin
            @(posedge clk);
            in_valid = 1;
            in_data = $signed(i);
        end

        @(posedge clk);
        in_valid = 0;

        wait(out_valid);
        @(posedge clk);

""" + (
        '        $display("  out_data = %0d", out_data);' if output_size == 1 else
        '''        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            $display("  out_data[%0d] = %0d", i, out_data[i]);
        end'''
    ) + """

        #100;
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid) $display("[%0t] out_valid asserted", $time);
    end

endmodule
"""
    out_path = out_dir / f"tb_{model_name}_top.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_mapping_report(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
    scale: int,
    data_width: int,
    weight_width: int,
    acc_width: int,
    frac_bits: int,
    *,
    final_activation: Optional[str] = None,
) -> Path:
    """Generate mapping_report.txt."""
    lines = []
    lines.append("RTL Mapping Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append(f"Scale Factor: {scale}")
    lines.append(f"Data Width: {data_width}")
    lines.append(f"Weight Width: {weight_width}")
    lines.append(f"Accumulator Width: {acc_width}")
    lines.append(f"Fractional Bits: {frac_bits}")
    if final_activation:
        lines.append(f"Final Activation (from ONNX): {final_activation}")
    lines.append("")
    # Add computed LAYER_SCALE and BIAS_SCALE for binaryclass format
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    if linear_layers:
        try:
            scales = compute_layer_scales_for_binaryclass(linear_layers, weight_width, scale)
            layer_scale_keys = sorted(k for k in scales if "LAYER_SCALE" in k)
            bias_scale_keys = sorted(k for k in scales if "BIAS_SCALE" in k)
            if layer_scale_keys:
                lines.append("Computed LAYER_SCALE (from dimensions + ONNX quant params):")
                for k in layer_scale_keys:
                    lines.append(f"  {k}: {scales[k]}")
            if bias_scale_keys:
                lines.append("Computed BIAS_SCALE (from accumulator/bias alignment):")
                for k in bias_scale_keys:
                    lines.append(f"  {k}: {scales[k]}")
            lines.append("")
        except Exception:
            pass
    # Per-layer activation (from ONNX graph trace)
    for layer in linear_layers:
        if layer.activation:
            lines.append(f"  {layer.name} activation: {layer.activation}")
    if any(l.activation for l in linear_layers):
        lines.append("")
    lines.append("Layer Mapping:")
    lines.append("-" * 80)

    for i, layer in enumerate(layers):
        line = f"{i+1:2d}. {layer.name:20s} {layer.layer_type:10s}"
        if layer.layer_type == "linear":
            line += f" in_features={layer.in_features:4d} out_features={layer.out_features:4d}"
            lines.append(line)
            lines.append(f"     Weights: mem_files/{layer.name}_weights.mem")
            lines.append(f"     Biases:  mem_files/{layer.name}_biases.mem")
        elif layer.layer_type == "relu":
            line += " (element-wise)"
            lines.append(line)
        elif layer.layer_type == "flatten":
            line += " (pass-through in streaming)"
            lines.append(line)
        else:
            lines.append(line)
        lines.append("")

    if final_activation:
        lines.append(f"  {len(layers) + 1:2d}. {(final_activation + '_final'):20s} {'activation':10s} (from ONNX graph)")
        lines.append("")

    lines.append("Generated Files:")
    lines.append("-" * 80)
    for layer in layers:
        if layer.layer_type == "linear":
            lines.append(f"  - mem_files/{layer.name}_weights.mem")
            lines.append(f"  - mem_files/{layer.name}_biases.mem")
    lines.append("")
    out_path = out_dir / "mapping_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def generate_netlist_json(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
    *,
    final_activation: Optional[str] = None,
) -> Path:
    """Generate netlist.json."""
    netlist = {
        "model_name": model_name,
        "layers": [],
        "final_activation": final_activation,
    }

    for layer in layers:
        layer_dict = {
            "name": layer.name,
            "type": layer.layer_type
        }
        if layer.layer_type == "linear":
            layer_dict["in_features"] = layer.in_features
            layer_dict["out_features"] = layer.out_features
            layer_dict["weight_mem"] = f"mem_files/{layer.name}_weights.mem"
            layer_dict["bias_mem"] = f"mem_files/{layer.name}_biases.mem"
        netlist["layers"].append(layer_dict)

    out_path = out_dir / "netlist.json"
    out_path.write_text(json.dumps(netlist, indent=2), encoding="utf-8")
    return out_path


# -----------------------------------------------------------------------------
# Layer Scale Computation (from ONNX quant params and/or dimensions)
# -----------------------------------------------------------------------------
#
# LAYER_SCALE: Right-shift on accumulator output to normalize sum-of-products
#   into output fractional scale. Depends on input_size (sum bit growth) and
#   optionally ONNX weight_scale (quantized weight scale).
#
# BIAS_SCALE: Left-shift on bias before adding to accumulator. Aligns bias
#   (stored in output scale) with accumulator (product scale + sum growth).
#

def compute_layer_scale(
    weight_width: int,
    input_size: int,
    weight_scale: Optional[float] = None,
    python_scale: int = 256,
) -> int:
   
    # Output fractional bits from global python_scale (e.g. 256 -> 8 bits)
    output_frac_bits = int(round(math.log2(python_scale))) if python_scale > 0 else 8
    # Bit growth from summing input_size products
    log2_n = max(0, math.ceil(math.log2(input_size)) if input_size > 0 else 0)
    if weight_scale is not None and weight_scale > 0:
        # ONNX weight_scale: product has scale weight_scale, so -log2(scale) fractional bits
        effective_weight_frac = max(0, math.ceil(-math.log2(weight_scale)))
        # Use the larger of output scale vs ONNX scale (avoid under-shifting)
        frac_bits = max(output_frac_bits, effective_weight_frac)
        shift = frac_bits + log2_n
    else:
        # No ONNX scale: use python_scale fractional bits only
        shift = output_frac_bits + log2_n
    return max(0, min(int(shift), 24))


def compute_bias_scale(
    weight_width: int,
    input_size: int,
    python_scale: int = 256,
    bias_scale_from_onnx: Optional[float] = None,
) -> int:

    # Bias is stored with python_scale fractional bits
    output_frac_bits = int(round(math.log2(python_scale))) if python_scale > 0 else 8
    log2_n = max(0, math.ceil(math.log2(input_size)) if input_size > 0 else 0)
    # Accumulator: product has 2*weight_width frac bits; sum adds log2(input_size) growth
    acc_frac = 2 * weight_width + log2_n
    bias_frac = output_frac_bits
    # Left-shift bias by (acc_frac - bias_frac) to align with accumulator
    scale = max(0, acc_frac - bias_frac)
    return min(scale, 16)


def compute_layer_scales_for_binaryclass(
    linear_layers: List[LayerInfo],
    weight_width: int,
    python_scale: int = 256,
) -> Dict[str, int]:
    """Compute LAYER_SCALE and BIAS_SCALE for each binaryclass block.
    Pairs layers consecutively: block b = (fc_{2b+1}, fc_{2b+2}); odd N: last block has (fcN, 1x1).
    IN layers use ONNX weight_scale when available; OUT layers use python_scale only."""
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    result: Dict[str, int] = {}

    def _ws(layer: LayerInfo) -> Optional[float]:
        if layer.quant_params and "weight_scale" in layer.quant_params:
            return layer.quant_params["weight_scale"]
        if layer.quant_params and "b_scale" in layer.quant_params:
            return layer.quant_params["b_scale"]
        return None

    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        in_layer = linear_layers[in_idx]
        out_layer = linear_layers[out_idx] if out_idx is not None else None
        in_f = in_layer.in_features or 0
        in_out = in_layer.out_features or 0
        rom_depth = (out_layer.out_features or 1) if out_layer else 1
        prefix = f"FC_{b + 1}"
        result[f"{prefix}_IN_LAYER_SCALE"] = compute_layer_scale(
            weight_width, in_f, _ws(in_layer), python_scale
        )
        result[f"{prefix}_IN_BIAS_SCALE"] = compute_bias_scale(weight_width, in_f, python_scale)
        result[f"{prefix}_OUT_LAYER_SCALE"] = compute_layer_scale(
            weight_width, rom_depth, None, python_scale
        )
        result[f"{prefix}_OUT_BIAS_SCALE"] = compute_bias_scale(weight_width, rom_depth, python_scale)

    return result


# -----------------------------------------------------------------------------
# Binaryclass_nn Format
# -----------------------------------------------------------------------------

def _compute_binaryclass_nn_params(linear_layers: List[LayerInfo], weight_width: int = 8, python_scale: int = 256) -> dict:
    """Compute binaryclass_NN parameters from ONNX layers.
    Pairs layers consecutively: block b = (fc_{2b+1}, fc_{2b+2}); odd N: last block has (fcN, 1x1).
    Returns dict with FC_1_*, FC_2_*, ... for num_blocks = ceil(N/2)."""
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    scales = compute_layer_scales_for_binaryclass(linear_layers, weight_width, python_scale)
    result: Dict[str, Any] = {"FIFO_DEPTH": 1024}
    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        in_layer = linear_layers[in_idx]
        out_layer = linear_layers[out_idx] if out_idx is not None else None
        in_f = in_layer.in_features or 0
        in_out = in_layer.out_features or 0
        rom_depth = (out_layer.out_features or 1) if out_layer else 1
        prefix = f"FC_{b + 1}"
        result[f"{prefix}_NEURONS"] = in_out
        result[f"{prefix}_INPUT_SIZE"] = in_f
        result[f"{prefix}_ROM_DEPTH"] = rom_depth
        result[f"{prefix}_IN_LAYER_SCALE"] = scales[f"{prefix}_IN_LAYER_SCALE"]
        result[f"{prefix}_IN_BIAS_SCALE"] = scales[f"{prefix}_IN_BIAS_SCALE"]
        result[f"{prefix}_OUT_LAYER_SCALE"] = scales[f"{prefix}_OUT_LAYER_SCALE"]
        result[f"{prefix}_OUT_BIAS_SCALE"] = scales[f"{prefix}_OUT_BIAS_SCALE"]
    return result


def _build_binaryclass_nn_sv_content(params: Dict[str, Any], num_blocks: int) -> str:
    """Build binaryclass_NN.sv module content for num_blocks (dynamic generation)."""
    param_lines = []
    for b in range(1, num_blocks + 1):
        p = f"FC_{b}"
        param_lines.append(f"  parameter int {p}_NEURONS          = 32'd{params.get(f'{p}_NEURONS', 1)},")
        param_lines.append(f"  parameter int {p}_INPUT_SIZE       = 32'd{params.get(f'{p}_INPUT_SIZE', 1)},")
        param_lines.append(f"  parameter int {p}_ROM_DEPTH        = 32'd{params.get(f'{p}_ROM_DEPTH', 1)},")
        param_lines.append(f"  parameter int {p}_IN_LAYER_SCALE   = 32'd{params.get(f'{p}_IN_LAYER_SCALE', 8)},")
        param_lines.append(f"  parameter int {p}_IN_BIAS_SCALE    = 32'd{params.get(f'{p}_IN_BIAS_SCALE', 0)},")
        param_lines.append(f"  parameter int {p}_OUT_LAYER_SCALE  = 32'd{params.get(f'{p}_OUT_LAYER_SCALE', 8)},")
        param_lines.append(f"  parameter int {p}_OUT_BIAS_SCALE   = 32'd{params.get(f'{p}_OUT_BIAS_SCALE', 0)},")
    param_lines.append(f"  parameter int FIFO_DEPTH            = 32'd{params.get('FIFO_DEPTH', 1024)}")
    param_str = "\n".join(param_lines)

    last_rom = params.get(f"FC_{num_blocks}_ROM_DEPTH", 1)
    sig_lines = []
    block_inst = []
    prev_valid = "s_axis_tvalid_i"
    prev_data = "s_axis_tdata_i"

    for b in range(1, num_blocks + 1):
        p = f"FC_{b}"
        neurons = params.get(f"{p}_NEURONS", 1)
        is_last = b == num_blocks
        sig_lines.append(f"  // Block {b} signals")
        sig_lines.append(f"  q_data_t fc{b}_out_s[{neurons}];")
        sig_lines.append(f"  logic    fc{b}_valid_s;")
        sig_lines.append(f"  q_data_t fc{b}_pre_act_s;")
        if not is_last:
            sig_lines.append(f"  q_data_t fc{b}_post_relu_s;")
            sig_lines.append(f"  logic    relu{b}_valid_i_s;")
            sig_lines.append(f"  logic    relu{b}_valid_o_s;")
        else:
            sig_lines.append(f"  logic    sigmoid_valid_i_s;")
            sig_lines.append(f"  logic    sigmoid_valid_o_s;")
            sig_lines.append(f"  q_data_t sigmoid_data_s;")
            sig_lines.append(f"  q_data_t logits_s;")
        sig_lines.append("")

        block_inst.append(f"  // Block {b}: fc_in + fc_out + activation")
        block_inst.append(f"  fc_in_layer #(")
        block_inst.append(f"    .NUM_NEURONS ({p}_NEURONS),")
        block_inst.append(f"    .INPUT_SIZE  ({p}_INPUT_SIZE),")
        block_inst.append(f"    .BIAS_SCALE  ({p}_IN_BIAS_SCALE),")
        block_inst.append(f"    .LAYER_SCALE ({p}_IN_LAYER_SCALE)")
        block_inst.append(f"  ) u_fc{b}_in_layer (")
        block_inst.append(f"    .clk_i   (clk_i),")
        block_inst.append(f"    .rst_n_i   (rst_n_i),")
        block_inst.append(f"    .valid_i ({prev_valid}),")
        block_inst.append(f"    .data_i  ({prev_data}),")
        block_inst.append(f"    .data_o  (fc{b}_out_s),")
        block_inst.append(f"    .valid_o (fc{b}_valid_s)")
        block_inst.append(f"  );")
        block_inst.append(f"  fc_out_layer #(")
        block_inst.append(f"    .NUM_NEURONS ({p}_NEURONS),")
        block_inst.append(f"    .ROM_DEPTH   ({p}_ROM_DEPTH),")
        block_inst.append(f"    .BIAS_SCALE  ({p}_OUT_BIAS_SCALE),")
        block_inst.append(f"    .LAYER_SCALE ({p}_OUT_LAYER_SCALE)")
        block_inst.append(f"  ) u_fc{b}_out_layer (")
        block_inst.append(f"    .clk_i   (clk_i),")
        block_inst.append(f"    .rst_n_i   (rst_n_i),")
        block_inst.append(f"    .valid_i (fc{b}_valid_s),")
        block_inst.append(f"    .data_i  (fc{b}_out_s),")
        block_inst.append(f"    .data_o  (fc{b}_pre_act_s),")
        block_inst.append(f"    .valid_o (relu{b}_valid_i_s)" if not is_last else f"    .valid_o (sigmoid_valid_i_s)")
        block_inst.append(f"  );")
        if not is_last:
            block_inst.append(f"  relu_layer u_relu{b} (")
            block_inst.append(f"    .clk_i   (clk_i),")
            block_inst.append(f"    .rst_n_i   (rst_n_i),")
            block_inst.append(f"    .valid_i (relu{b}_valid_i_s),")
            block_inst.append(f"    .data_i  (fc{b}_pre_act_s),")
            block_inst.append(f"    .data_o  (fc{b}_post_relu_s),")
            block_inst.append(f"    .valid_o (relu{b}_valid_o_s)")
            block_inst.append(f"  );")
            prev_valid = f"relu{b}_valid_o_s"
            prev_data = f"fc{b}_post_relu_s"
        else:
            block_inst.append(f"  sigmoid_layer u_sigmoid_layer (")
            block_inst.append(f"    .clk_i   (clk_i),")
            block_inst.append(f"    .rst_n_i   (rst_n_i),")
            block_inst.append(f"    .valid_i (sigmoid_valid_i_s),")
            block_inst.append(f"    .data_i  (fc{b}_pre_act_s),")
            block_inst.append(f"    .data_o  (sigmoid_data_s),")
            block_inst.append(f"    .valid_o (sigmoid_valid_o_s)")
            block_inst.append(f"  );")
        block_inst.append("")

    sig_lines.append("  logic fifo_empty_s;")
    sig_lines.append("  logic fifo_empty_q;")
    sig_lines.append("  logic fifo_full_s;")
    sig_lines.append("  logic fifo_write_en_s;")
    sig_lines.append("  logic fifo_read_en_s;")
    sig_lines.append("  logic fifo_read_en_q;")
    sig_lines.append("  logic [DATA_WIDTH-1:0] fifo_read_data_s;")
    sig_lines.append("  logic [DATA_WIDTH-1:0] fifo_write_data_s;")
    sig_lines.append(f"  logic [$clog2({last_rom} + 1) - 1:0] out_count_q;")
    sig_lines.append("  logic tvalid_q;")

    body = f'''`begin_keywords "1800-2012"
module binaryclass_NN 
  import quant_pkg::*;
#(
{param_str}
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  output logic [DATA_WIDTH-1:0]       m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0]       m_axis_prediction_tkeep_o,
  output logic                        m_axis_prediction_tvalid_o,
  input  logic                        m_axis_prediction_tready_i,
  output logic                        m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

{"".join(s + chr(10) for s in sig_lines)}

{"".join(s + chr(10) for s in block_inst)}

  sync_fifo #(
    .DATA_WIDTH(DATA_WIDTH),
    .DEPTH     (FIFO_DEPTH)     
) u_fifo_sigmoid (
    .clk_i        (clk_i),
    .rst_n_i        (rst_n_i),
    .write_en_i   (fifo_write_en_s),
    .write_data_i (fifo_write_data_s),
    .full_o       (fifo_full_s),
    .read_en_i   (fifo_read_en_s),
    .read_data_o (fifo_read_data_s),
    .empty_o     (fifo_empty_s)
);

always_ff @(posedge clk_i or negedge rst_n_i) begin : fifo_read_pipe
  if (!rst_n_i)
    fifo_read_en_q <= 1'b0;
  else
    fifo_read_en_q <= fifo_read_en_s;
end : fifo_read_pipe

always_ff @(posedge clk_i or negedge rst_n_i) begin : axi_output_logic
  if (!rst_n_i) begin
    m_axis_prediction_tdata_o <= 32'h0;
    tvalid_q                <= 1'b0;
  end
  else if (fifo_read_en_q) begin
    m_axis_prediction_tdata_o <= fifo_read_data_s;
    tvalid_q                <= 1'b1;
  end
  else if (m_axis_prediction_tready_i) begin
    tvalid_q <= 1'b0;
  end
end : axi_output_logic

always_ff @(posedge clk_i or negedge rst_n_i) begin : output_tracking
  if (!rst_n_i) begin
    out_count_q <= '0;
  end
  else if (m_axis_prediction_tvalid_o && m_axis_prediction_tready_i) begin
    if (out_count_q == ({last_rom} - 1))
      out_count_q <= '0;
    else
      out_count_q <= out_count_q + 1'b1;
  end
end : output_tracking

assign fifo_write_en_s   = sigmoid_valid_o_s && !fifo_full_s;
assign fifo_read_en_s    = !fifo_empty_s && m_axis_prediction_tready_i;
assign fifo_write_data_s = {{24'h0, sigmoid_data_s}};

assign m_axis_prediction_tvalid_o = tvalid_q;
assign m_axis_prediction_tkeep_o  = 4'h1;
assign m_axis_prediction_tlast_o = (m_axis_prediction_tvalid_o && (out_count_q == ({last_rom} - 1)));

endmodule : binaryclass_NN
`end_keywords
'''
    return body


def generate_binaryclass_NN_top_from_template(
    out_dir: Path,
    linear_layers: List[LayerInfo],
    weight_width: int = 8,
    python_scale: int = 256,
) -> Path:
    """Generate binaryclass_NN.sv dynamically for any number of layers (paired into blocks)."""
    params = _compute_binaryclass_nn_params(linear_layers, weight_width, python_scale)
    num_blocks = (len(linear_layers) + 1) // 2
    content = _build_binaryclass_nn_sv_content(params, num_blocks)
    out_path = out_dir / "binaryclass_NN.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    LOGGER.info(f"  Generated binaryclass_NN.sv (params from ONNX, {num_blocks} blocks)")
    return out_path


def generate_binaryclass_NN_wrapper_from_template(
    out_dir: Path,
    linear_layers: List[LayerInfo],
    weight_width: int = 8,
    python_scale: int = 256,
) -> Path:
    """Generate binaryclass_NN_wrapper.sv dynamically for any number of layers."""
    params = _compute_binaryclass_nn_params(linear_layers, weight_width, python_scale)
    num_blocks = (len(linear_layers) + 1) // 2
    param_lines = []
    for b in range(1, num_blocks + 1):
        p = f"FC_{b}"
        param_lines.append(f"    .{p}_NEURONS    (FC{b}_NEURONS),")
        param_lines.append(f"    .{p}_INPUT_SIZE (FC{b}_INPUT_SIZE),")
        param_lines.append(f"    .{p}_ROM_DEPTH  (FC{b}_ROM_DEPTH),")
        param_lines.append(f"    .{p}_IN_LAYER_SCALE   ({params.get(f'{p}_IN_LAYER_SCALE', 8)}),")
        param_lines.append(f"    .{p}_IN_BIAS_SCALE    ({params.get(f'{p}_IN_BIAS_SCALE', 0)}),")
        param_lines.append(f"    .{p}_OUT_LAYER_SCALE  ({params.get(f'{p}_OUT_LAYER_SCALE', 8)}),")
        param_lines.append(f"    .{p}_OUT_BIAS_SCALE   ({params.get(f'{p}_OUT_BIAS_SCALE', 0)})")
        if b < num_blocks:
            param_lines[-1] += ","
    wrapper_params_str = "\n".join(param_lines)
    wrapper_param_decls = []
    for b in range(1, num_blocks + 1):
        wrapper_param_decls.append(f"  parameter int FC{b}_NEURONS    = {params.get(f'FC_{b}_NEURONS', 1)},")
        wrapper_param_decls.append(f"  parameter int FC{b}_INPUT_SIZE = {params.get(f'FC_{b}_INPUT_SIZE', 1)},")
        wrapper_param_decls.append(f"  parameter int FC{b}_ROM_DEPTH  = {params.get(f'FC_{b}_ROM_DEPTH', 1)}")
        if b < num_blocks:
            wrapper_param_decls[-1] += ","

    body = f'''`begin_keywords "1800-2012"
module binaryclass_NN_wrapper 
  import quant_pkg::*;
#(
{"".join(wrapper_param_decls)}
)(
  input  logic clk_i,
  input  logic rst_n_i,

  input  q_data_t s_axis_tdata_i,
  input  logic    s_axis_tvalid_i,
  input  logic    s_axis_tlast_i,

  output logic [DATA_WIDTH-1:0]       m_axis_prediction_tdata_o,
  output logic [KEEP_WIDTH-1:0]       m_axis_prediction_tkeep_o,
  output logic                        m_axis_prediction_tvalid_o,
  input  logic                        m_axis_prediction_tready_i,
  output logic                        m_axis_prediction_tlast_o
);

  timeunit 1ns;
  timeprecision 1ps;

  binaryclass_NN #(
{wrapper_params_str}
  ) u_binaryclass_nn (
    .clk_i   (clk_i),
    .rst_n_i   (rst_n_i),
    .s_axis_tdata_i                  (s_axis_tdata_i),
    .s_axis_tvalid_i                 (s_axis_tvalid_i),
    .s_axis_tlast_i                  (s_axis_tlast_i),
    .m_axis_prediction_tdata_o       (m_axis_prediction_tdata_o),
    .m_axis_prediction_tkeep_o       (m_axis_prediction_tkeep_o),
    .m_axis_prediction_tvalid_o      (m_axis_prediction_tvalid_o),
    .m_axis_prediction_tready_i      (m_axis_prediction_tready_i),
    .m_axis_prediction_tlast_o       (m_axis_prediction_tlast_o)
  );

endmodule : binaryclass_NN_wrapper
`end_keywords
'''
    out_path = out_dir / "binaryclass_NN_wrapper.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body.rstrip() + "\n", encoding="utf-8")
    LOGGER.info(f"  Generated binaryclass_NN_wrapper.sv")
    return out_path


def emit_binaryclass_nn_format(
    out_dir: Path,
    layers: List[LayerInfo],
    input_size: int,
    weight_width: int,
    scale: int,
) -> None:
    """Emit RTL in binaryclass_nn format: binaryclass_NN module, AXI4-Stream ports.
    Supports any number of FC layers; pairs consecutively into ceil(N/2) blocks."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    if not linear_layers:
        raise RuntimeError("binaryclass_nn format requires at least 1 FC layer.")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = out_dir

    # 1. Write reusable templates
    (out_dir / "quant_pkg.sv").write_text(_get_quant_pkg_from_template(weight_width), encoding="utf-8")
    _write_template_or_embedded(out_dir, "mac.sv", _load_reusable_template("mac.sv"), BINARYCLASS_MAC_SV)
    _write_template_or_embedded(out_dir, "fc_in.sv", _load_reusable_template("fc_in.sv"), BINARYCLASS_FC_IN_SV)
    _write_template_or_embedded(out_dir, "fc_out.sv", _load_reusable_template("fc_out.sv"), BINARYCLASS_FC_OUT_SV)
    _write_template_or_embedded(out_dir, "relu_layer.sv", _load_reusable_template("relu_layer.sv"), BINARYCLASS_RELU_LAYER_SV)
    _write_template_or_embedded(out_dir, "sigmoid_layer.sv", _load_reusable_template("sigmoid_layer.sv"), BINARYCLASS_SIGMOID_LAYER_SV)
    _write_template_or_embedded(out_dir, "sync_fifo.sv", _load_reusable_template("sync_fifo.sv"), BINARYCLASS_SYNC_FIFO_SV)
    LOGGER.info("  Wrote reusable templates from binaryclass_nn (quant_pkg, mac, fc_in, fc_out, relu_layer, sigmoid_layer, sync_fifo)")

    # 2. Generate .mem files (paired: block b = (fc_{2b-1}, fc_{2b}); odd N: last block has (fcN, 1x1))
    generate_proj_mem_files(layers, mem_dir, scale, weight_width)

    # 3. Generate ROMs for each block
    n = len(linear_layers)
    num_blocks = (n + 1) // 2
    params = _compute_binaryclass_nn_params(linear_layers, weight_width, scale)
    input_sizes = [params[f"FC_{b}_INPUT_SIZE"] for b in range(1, num_blocks + 1)]
    rom_depths = [params[f"FC_{b}_ROM_DEPTH"] for b in range(1, num_blocks + 1)]

    for b in range(num_blocks):
        in_idx = 2 * b
        out_idx = 2 * b + 1 if 2 * b + 1 < n else None
        in_layer = linear_layers[in_idx]
        out_layer = linear_layers[out_idx] if out_idx is not None else None
        in_f = in_layer.in_features or 0
        in_out = in_layer.out_features or 0
        rom_d = (out_layer.out_features or 1) if out_layer else 1
        proj_out_f = (out_layer.out_features or 1) if out_layer else 1
        prefix = _proj_prefix(b)
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_in_weights_rom", in_f, in_out, weight_width, mem_dir, "weights")
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_in_bias_rom", 1, in_out, weight_width, mem_dir, "biases")
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_out_weights_rom", rom_d, proj_out_f, weight_width, mem_dir, "weights")
        _generate_binaryclass_nn_rom(out_dir, f"{prefix}_out_bias_rom", rom_d, 1, weight_width, mem_dir, "biases")

    # 4. Generate fc_in_layer and fc_out_layer (generate blocks for each block's INPUT_SIZE / ROM_DEPTH)
    _generate_fc_in_layer_binaryclass(out_dir, input_sizes, weight_width, mem_dir)
    _generate_fc_out_layer_binaryclass(out_dir, rom_depths, weight_width, mem_dir)

    # 5. Generate binaryclass_NN and wrapper
    generate_binaryclass_NN_top_from_template(out_dir, linear_layers, weight_width, scale)
    generate_binaryclass_NN_wrapper_from_template(out_dir, linear_layers, weight_width, scale)


def _generate_binaryclass_nn_rom(
    out_dir: Path,
    rom_name: str,
    depth: int,
    width_neurons: int,
    weight_width: int,
    mem_dir: Path,
    mem_type: str,
    *,
    mem_path_prefix: str = "",
) -> None:
    """Generate a ROM module with binaryclass_nn naming. mem_path_prefix: '' for root layout.
    rom_name must be e.g. fc_proj_in_weights_rom or fc_proj_in_bias_rom (matches fc_in_layer/fc_out_layer)."""
    base = rom_name.replace("_weights_rom", "").replace("_bias_rom", "")
    if "bias" in rom_name:
        mem_file = f"{mem_path_prefix}{base}_bias.mem"
    else:
        mem_file = f"{mem_path_prefix}{base}_weights.mem"
    packed_width = width_neurons * weight_width
    if "bias" in rom_name and "in" in rom_name:
        packed_width = width_neurons * weight_width * 4  # acc_t per neuron for fc_in bias
    elif "bias" in rom_name and "out" in rom_name:
        packed_width = weight_width * 4  # acc_t for fc_out bias
    body = f"""module {rom_name} #(
  parameter int DEPTH  = {depth},
  parameter int WIDTH  = {packed_width},
  parameter int ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
  input  logic              clk_i,
  input  logic [ADDR_W-1:0] addr_i,
  output logic [WIDTH-1:0]  data_o
);

  timeunit 1ns;
  timeprecision 1ps;

  (* rom_style = "block" *)
  logic [WIDTH-1:0] mem [0:DEPTH-1];

  initial begin
    $readmemh("{mem_file}", mem);
  end

  always_ff @(posedge clk_i) begin : read_port
    data_o <= mem[addr_i];
  end : read_port

endmodule : {rom_name}
"""
    (out_dir / f"{rom_name}.sv").write_text(_pyramidtech_wrap(body, f"{rom_name}.sv", f"ROM for {rom_name}"), encoding="utf-8")


def _generate_fc_in_layer_binaryclass(out_dir: Path, input_sizes: List[int], weight_width: int, mem_dir: Path) -> None:
    """Generate fc_in_layer with generate blocks for each block's INPUT_SIZE (ROM selection)."""
    branches = []
    for b, in_sz in enumerate(input_sizes):
        prefix = _proj_prefix(b)
        cond = "if" if b == 0 else "else if"
        branches.append(f"""        {cond} (INPUT_SIZE == {in_sz}) begin
            {prefix}_in_weights_rom #(.DEPTH(INPUT_SIZE), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk_i), .addr_i(weight_addr), .data_o(weight_rom_row));
            {prefix}_in_bias_rom #(.DEPTH(1), .WIDTH(NUM_NEURONS*Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk_i), .addr_i(1'b0), .data_o(bias_rom_row));
        end""")
    gen_block = "\n".join(branches)
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_in_layer #(
    parameter int NUM_NEURONS = 8,
    parameter int INPUT_SIZE  = 16,
    parameter signed LAYER_SCALE = 12,
    parameter signed BIAS_SCALE  = 0
)(
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i,

    output q_data_t data_o[NUM_NEURONS],
    output logic valid_o
);

    logic [$clog2(INPUT_SIZE)-1:0] weight_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weight_rom_row;
    logic [NUM_NEURONS*Q_WIDTH*4-1:0] bias_rom_row;
    q_data_t weights_rom_data[NUM_NEURONS];
    acc_t bias_rom_data[NUM_NEURONS];
    logic valid_i_q;
    q_data_t data_i_q;

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            valid_i_q <= 1'b0;
            data_i_q  <= '0;
        end else begin
            valid_i_q <= valid_i;
            data_i_q  <= data_i;
        end
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            weight_addr <= '0;
        else if (valid_i)
            weight_addr <= (weight_addr == INPUT_SIZE-1) ? '0 : weight_addr + 1'b1;
    end

    genvar n;
    generate
{gen_block}
    endgenerate

    generate
        for (n = 0; n < NUM_NEURONS; n = n + 1) begin : SPLIT_ROM
            assign weights_rom_data[n] = weight_rom_row[n*Q_WIDTH +: Q_WIDTH];
            assign bias_rom_data[n]    = bias_rom_row[n*Q_WIDTH*4 +: Q_WIDTH*4];
        end
    endgenerate

    fc_in #(
        .NUM_NEURONS(NUM_NEURONS),
        .INPUT_SIZE(INPUT_SIZE),
        .LAYER_SCALE(LAYER_SCALE),
        .BIAS_SCALE(BIAS_SCALE)
    ) u_fc_in (
        .clk_i(clk_i),
        .rst_n_i(rst_n_i),
        .valid_i(valid_i_q),
        .data_i(data_i_q),
        .weights_i(weights_rom_data),
        .biases_i(bias_rom_data),
        .data_o(data_o),
        .valid_o(valid_o)
    );

endmodule
"""
    (out_dir / "fc_in_layer.sv").write_text(_pyramidtech_wrap(body, "fc_in_layer.sv", "FC input layer with ROM"), encoding="utf-8")


def _generate_fc_out_layer_binaryclass(out_dir: Path, rom_depths: List[int], weight_width: int, mem_dir: Path) -> None:
    """Generate fc_out_layer with generate blocks for each block's ROM_DEPTH."""
    branches = []
    for b, rom_d in enumerate(rom_depths):
        prefix = _proj_prefix(b)
        cond = "if" if b == 0 else "else if"
        addr_arg = "1'b0" if rom_d <= 1 else "weights_rom_addr"
        bias_addr = "1'b0" if rom_d <= 1 else "bias_rom_addr"
        branches.append(f"""        {cond} (ROM_DEPTH == {rom_d}) begin
            {prefix}_out_weights_rom #(.DEPTH(ROM_DEPTH), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk_i), .addr_i({addr_arg}), .data_o(weights_rom_data_raw));
            {prefix}_out_bias_rom #(.DEPTH(ROM_DEPTH), .WIDTH(Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk_i), .addr_i({bias_addr}), .data_o(bias_rom_data));
        end""")
    gen_block = "\n".join(branches)
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module fc_out_layer #(
    parameter int NUM_NEURONS = 2,
    parameter int ROM_DEPTH   = 45,
    parameter signed LAYER_SCALE = 5,
    parameter signed BIAS_SCALE  = 1
)(
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic valid_i,
    input  q_data_t data_i[NUM_NEURONS],

    output q_data_t data_o,
    output logic valid_o
);

    logic [$clog2(ROM_DEPTH)-1:0] addr_cnt;
    logic addr_last;
    logic valid_pipeline;
    logic valid_pipeline_reg;
    logic valid_out_engine;
    logic valid_out_engine_reg;
    logic [$clog2(ROM_DEPTH)-1:0] weights_rom_addr;
    logic [$clog2(ROM_DEPTH)-1:0] bias_rom_addr;
    logic [NUM_NEURONS*Q_WIDTH-1:0] weights_rom_data_raw;
    q_data_t weights_rom_data[NUM_NEURONS];
    acc_t bias_rom_data;
    q_data_t weights_rom_data_reg[NUM_NEURONS];
    acc_t bias_rom_data_reg;
    q_data_t data_i_reg[NUM_NEURONS];

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            addr_cnt <= '0;
        else if (valid_pipeline)
            addr_cnt <= (addr_cnt == ROM_DEPTH-1) ? '0 : addr_cnt + 1'b1;
    end

    assign addr_last = (addr_cnt == ROM_DEPTH-1);

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i)
            valid_pipeline <= 1'b0;
        else if (valid_i)
            valid_pipeline <= 1'b1;
        else if (addr_last)
            valid_pipeline <= 1'b0;
    end

    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            valid_pipeline_reg   <= 1'b0;
            valid_out_engine_reg <= 1'b0;
        end else begin
            valid_pipeline_reg   <= valid_pipeline;
            valid_out_engine_reg <= valid_out_engine;
        end
    end

    assign valid_o = valid_out_engine_reg;
    assign weights_rom_addr = addr_cnt;
    assign bias_rom_addr    = addr_cnt;

    generate
{gen_block}
    endgenerate

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : SPLIT_WEIGHTS
            assign weights_rom_data[i] = weights_rom_data_raw[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    integer k;
    always_ff @(posedge clk_i or negedge rst_n_i) begin
        if (!rst_n_i) begin
            for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                weights_rom_data_reg[k] <= '0;
                data_i_reg[k] <= '0;
            end
            bias_rom_data_reg <= '0;
        end else begin
            for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                weights_rom_data_reg[k] <= weights_rom_data[k];
                data_i_reg[k] <= data_i[k];
            end
            bias_rom_data_reg <= bias_rom_data;
        end
    end

    fc_out #(
        .NUM_NEURONS(NUM_NEURONS),
        .LAYER_SCALE(LAYER_SCALE),
        .BIAS_SCALE(BIAS_SCALE)
    ) u_fc_out (
        .clk_i(clk_i),
        .rst_n_i(rst_n_i),
        .valid_i(valid_pipeline_reg),
        .data_i(data_i_reg),
        .weights_i(weights_rom_data_reg),
        .bias_i(bias_rom_data_reg),
        .data_o(data_o),
        .valid_o(valid_out_engine)
    );

endmodule
"""
    (out_dir / "fc_out_layer.sv").write_text(_pyramidtech_wrap(body, "fc_out_layer.sv", "FC output layer with ROM"), encoding="utf-8")


def _generate_binaryclass_NN(
    out_dir: Path,
    fc1_n: int, fc2_n: int, fc3_n: int,
    fc1_in: int, fc2_in: int, fc3_in: int,
    fc1_rom: int, fc2_rom: int, fc3_rom: int,
) -> None:
    """Generate binaryclass_NN.sv with AXI4-Stream ports matching reference."""
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module binaryclass_NN #(
    parameter int FC1_NEURONS    = {fc1_n},
    parameter int FC2_NEURONS    = {fc2_n},
    parameter int FC3_NEURONS    = {fc3_n},
    parameter int FC1_INPUT_SIZE = {fc1_in},
    parameter int FC2_INPUT_SIZE = {fc2_in},
    parameter int FC3_INPUT_SIZE = {fc3_in},
    parameter int FC1_ROM_DEPTH  = {fc1_rom},
    parameter int FC2_ROM_DEPTH  = {fc2_rom},
    parameter int FC3_ROM_DEPTH  = {fc3_rom}
)(
    input  logic clk,
    input  logic rst_n,

    input  q_data_t axi_in_tdata,
    input  logic    axi_in_tvalid,
    input  logic    axi_in_tlast,

    output q_data_t prediction,
    output logic    prediction_valid,
    output logic    prediction_tlast
);

    q_data_t fc1_out[FC1_NEURONS];
    logic    fc1_layer_valid_out;

    q_data_t fc1_pre_relu;
    q_data_t fc1_post_relu;
    logic    fc1_relu_valid_in;
    logic    fc1_relu_valid_out;

    q_data_t fc2_out[FC2_NEURONS];
    logic    fc2_layer_valid_out;

    q_data_t fc2_pre_relu;
    q_data_t fc2_post_relu;
    logic    fc2_relu_valid_in;
    logic    fc2_relu_valid_out;

    q_data_t fc3_out[FC3_NEURONS];
    logic    fc3_layer_valid_out;

    logic    sigmoid_valid_in;
    q_data_t logits;

    logic [$clog2(FC3_ROM_DEPTH)-1:0] out_count;

    fc_in_layer #(.NUM_NEURONS(FC1_NEURONS), .INPUT_SIZE(FC1_INPUT_SIZE), .BIAS_SCALE(0), .LAYER_SCALE(12))
    fc1_in_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(axi_in_tvalid),
        .data_in(axi_in_tdata),
        .data_out(fc1_out),
        .valid_out(fc1_layer_valid_out)
    );

    fc_out_layer #(.NUM_NEURONS(FC1_NEURONS), .ROM_DEPTH(FC1_ROM_DEPTH), .LAYER_SCALE(7), .BIAS_SCALE(0))
    fc1_out_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_layer_valid_out),
        .data_in(fc1_out),
        .data_out(fc1_pre_relu),
        .valid_out(fc1_relu_valid_in)
    );

    relu_layer relu1 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_relu_valid_in),
        .data_in(fc1_pre_relu),
        .data_out(fc1_post_relu),
        .valid_out(fc1_relu_valid_out)
    );

    fc_in_layer #(.NUM_NEURONS(FC2_NEURONS), .INPUT_SIZE(FC2_INPUT_SIZE), .BIAS_SCALE(0), .LAYER_SCALE(7))
    fc2_in_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc1_relu_valid_out),
        .data_in(fc1_post_relu),
        .data_out(fc2_out),
        .valid_out(fc2_layer_valid_out)
    );

    fc_out_layer #(.NUM_NEURONS(FC2_NEURONS), .ROM_DEPTH(FC2_ROM_DEPTH), .LAYER_SCALE(7), .BIAS_SCALE(0))
    fc2_out_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_layer_valid_out),
        .data_in(fc2_out),
        .data_out(fc2_pre_relu),
        .valid_out(fc2_relu_valid_in)
    );

    relu_layer relu2 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_relu_valid_in),
        .data_in(fc2_pre_relu),
        .data_out(fc2_post_relu),
        .valid_out(fc2_relu_valid_out)
    );

    fc_in_layer #(.NUM_NEURONS(FC3_NEURONS), .INPUT_SIZE(FC3_INPUT_SIZE), .BIAS_SCALE(0), .LAYER_SCALE(7))
    fc3_in_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc2_relu_valid_out),
        .data_in(fc2_post_relu),
        .data_out(fc3_out),
        .valid_out(fc3_layer_valid_out)
    );

    fc_out_layer #(.NUM_NEURONS(FC3_NEURONS), .ROM_DEPTH(FC3_ROM_DEPTH), .LAYER_SCALE(6), .BIAS_SCALE(0))
    fc3_out_layer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(fc3_layer_valid_out),
        .data_in(fc3_out),
        .data_out(logits),
        .valid_out(sigmoid_valid_in)
    );

    sigmoid_layer sigmoid_layer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(logits),
        .valid_in(sigmoid_valid_in),
        .data_out(prediction),
        .valid_out(prediction_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            out_count <= 0;
        else if (prediction_valid) begin
            if (out_count == FC3_ROM_DEPTH-1)
                out_count <= 0;
            else
                out_count <= out_count + 1;
        end
    end

    assign prediction_tlast = (prediction_valid && (out_count == FC3_ROM_DEPTH-1));

endmodule
"""
    (out_dir / "binaryclass_NN.sv").write_text(_pyramidtech_wrap(body, "binaryclass_NN.sv", "Top-level binary classification NN with AXI4-Stream"), encoding="utf-8")


def _generate_binaryclass_NN_wrapper(
    out_dir: Path,
    fc1_n: int, fc2_n: int, fc3_n: int,
    fc1_in: int, fc2_in: int, fc3_in: int,
    fc1_rom: int, fc2_rom: int, fc3_rom: int,
) -> None:
    """Generate binaryclass_NN_wrapper.sv - adapts AXI-style ports to binaryclass_NN."""
    body = f"""`timescale 1ns/1ps
import quant_pkg::*;

module binaryclass_NN_wrapper #(
    parameter int FC1_NEURONS    = {fc1_n},
    parameter int FC2_NEURONS   = {fc2_n},
    parameter int FC3_NEURONS   = {fc3_n},
    parameter int FC1_INPUT_SIZE = {fc1_in},
    parameter int FC2_INPUT_SIZE = {fc2_in},
    parameter int FC3_INPUT_SIZE = {fc3_in},
    parameter int FC1_ROM_DEPTH  = {fc1_rom},
    parameter int FC2_ROM_DEPTH  = {fc2_rom},
    parameter int FC3_ROM_DEPTH  = {fc3_rom}
)(
    input  logic clk,
    input  logic rst_n,

    input  q_data_t s_axis_tdata_i,
    input  logic    s_axis_tvalid_i,
    input  logic    s_axis_tlast_i,

    output logic [31:0] m_axis_prediction_tdata_o,
    output logic [3:0]  m_axis_prediction_tkeep_o,
    output logic        m_axis_prediction_tvalid_o,
    input  logic        m_axis_prediction_tready_i,
    output logic        m_axis_prediction_tlast_o
);

    q_data_t prediction;

    binaryclass_NN #(
        .FC1_NEURONS(FC1_NEURONS), .FC2_NEURONS(FC2_NEURONS), .FC3_NEURONS(FC3_NEURONS),
        .FC1_INPUT_SIZE(FC1_INPUT_SIZE), .FC2_INPUT_SIZE(FC2_INPUT_SIZE), .FC3_INPUT_SIZE(FC3_INPUT_SIZE),
        .FC1_ROM_DEPTH(FC1_ROM_DEPTH), .FC2_ROM_DEPTH(FC2_ROM_DEPTH), .FC3_ROM_DEPTH(FC3_ROM_DEPTH)
    ) u_binaryclass_nn (
        .clk(clk),
        .rst_n(rst_n),
        .axi_in_tdata(s_axis_tdata_i),
        .axi_in_tvalid(s_axis_tvalid_i),
        .axi_in_tlast(s_axis_tlast_i),
        .prediction(prediction),
        .prediction_valid(m_axis_prediction_tvalid_o),
        .prediction_tlast(m_axis_prediction_tlast_o)
    );

    assign m_axis_prediction_tdata_o = {{(32-Q_WIDTH){{1'b0}}, prediction}};
    assign m_axis_prediction_tkeep_o = 4'h1;

endmodule
"""
    (out_dir / "binaryclass_NN_wrapper.sv").write_text(_pyramidtech_wrap(body, "binaryclass_NN_wrapper.sv", "Wrapper for binaryclass_NN"), encoding="utf-8")


def generate_rtl_filelist(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
    *,
    binaryclass_nn_format: bool = False,
    parameterized_layers: bool = False,
) -> Path:
    """Generate rtl_filelist.f in binaryclass_nn or hierarchical format."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    root_var = "$ROOT_DIR"
    incdir = f"+incdir+{root_var}/\n"
    lines = [incdir]
    if binaryclass_nn_format:
        files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "fc_in_layer.sv", "fc_out_layer.sv", "relu_layer.sv", "sigmoid_layer.sv", "sync_fifo.sv"]
        num_blocks = (len(linear_layers) + 1) // 2
        for b in range(num_blocks):
            prefix = _proj_prefix(b)
            files.extend([f"{prefix}_in_weights_rom.sv", f"{prefix}_in_bias_rom.sv", f"{prefix}_out_weights_rom.sv", f"{prefix}_out_bias_rom.sv"])
        files.append("binaryclass_NN.sv")
        files.append("binaryclass_NN_wrapper.sv")
        top_file = "binaryclass_NN_wrapper.sv"
    elif parameterized_layers:
        files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "relu_layer.sv", "relu_layer_array.sv", "sigmoid_layer.sv", "sync_fifo.sv", "fc_in_layer.sv", "fc_out_layer.sv"]
        for idx in range(len(linear_layers)):
            prefix = _proj_prefix(idx)
            files.append(f"{prefix}_in_weights_rom.sv")
            files.append(f"{prefix}_in_bias_rom.sv")
            files.append(f"{prefix}_out_weights_rom.sv")
            files.append(f"{prefix}_out_bias_rom.sv")
        files.append(f"{model_name}_top.sv")
        files.append(f"{model_name}_axi4_wrapper.sv")
        top_file = f"{model_name}_axi4_wrapper.sv"
    else:
        files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "relu_layer.sv", "relu_layer_array.sv"]
        for layer in linear_layers:
            files.append(f"weight_rom_{layer.name}.sv")
            files.append(f"bias_rom_{layer.name}.sv")
        for layer in linear_layers:
            if layer.name == "fc1":
                files.append(f"fc_layer_{layer.name}.sv")
            else:
                files.append(f"fc_out_layer_{layer.name}.sv")
        files.append(f"{model_name}_top.sv")
        files.append(f"{model_name}_axi4_wrapper.sv")
        top_file = f"{model_name}_axi4_wrapper.sv"
    for f in files:
        lines.append(f"{root_var}/{f}\n")
    out_path = out_dir / "rtl_filelist.f"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path