#!/usr/bin/env python3
"""
Minimal rtl_mapper for binary_onnx_to_rtl.py.
Contains only the functions needed for ONNX-to-RTL conversion (no PyTorch model loading).
Output format matches binaryclass_nn: PyramidTech header, clk_i/rst_n_i, _s/_q suffixes, etc.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

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
) -> None:
    """Generate .mem files with fc_N_proj_in/out naming for parameterized hierarchical flow."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    for idx, layer in enumerate(linear_layers):
        prefix = _proj_prefix(idx)
        w_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        b_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((layer.out_features or 0,), dtype=np.float32)
        wq = float_to_int(w_np, scale, bit_width)
        bq = float_to_int(b_np, scale, bit_width)
        in_f = layer.in_features or 0
        out_f = layer.out_features or 0

        (mem_dir / f"{prefix}_in_weights.mem").parent.mkdir(parents=True, exist_ok=True)
        generate_quant_pkg_style_weight_mem(wq, mem_dir / f"{prefix}_in_weights.mem", "fc1", in_f, out_f, bit_width)
        generate_proj_bias_mem_acc(bq, mem_dir / f"{prefix}_in_bias.mem", out_f, bit_width)

        if idx + 1 < len(linear_layers):
            next_layer = linear_layers[idx + 1]
            next_w = next_layer.weight.detach().cpu().numpy().astype(np.float32)
            next_b = next_layer.bias.detach().cpu().numpy().astype(np.float32) if next_layer.bias is not None else np.zeros((next_layer.out_features or 0,), dtype=np.float32)
            next_wq = float_to_int(next_w, scale, bit_width)
            next_bq = float_to_int(next_b, scale, bit_width)
            next_in = next_layer.in_features or 0
            next_out = next_layer.out_features or 0
            generate_quant_pkg_style_weight_mem(next_wq, mem_dir / f"{prefix}_out_weights.mem", "fc2", next_in, next_out, bit_width)
            generate_proj_out_bias_mem(next_bq, mem_dir / f"{prefix}_out_bias.mem", bit_width)


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
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic clear,
    input  q_data_t a,
    input  q_data_t b,
    output acc_t acc,
    output logic valid_out
);

    q_mult_t mult_reg;
    acc_t acc_reg;
    logic enable_q;
    logic enable_qq;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            mult_reg <= '0;
        else if (clear)
            mult_reg <= '0;
        else if (enable)
            mult_reg <= $signed(a) * $signed(b);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= '0;
        else if (clear)
            acc_reg <= '0;
        else if (enable_q)
            acc_reg <= acc_reg + mult_reg;
    end

    always_ff @(posedge clk) begin
        if (acc_reg > $signed(ACC_FULL_MAX))
            acc <= ACC_FULL_MAX;
        else if (acc_reg < $signed(ACC_FULL_MIN))
            acc <= ACC_FULL_MIN;
        else
            acc <= acc_reg[4*Q_WIDTH-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            enable_q  <= 1'b0;
            enable_qq <= 1'b0;
            valid_out <= 1'b0;
        end else if (clear) begin
            enable_q  <= 1'b0;
            enable_qq <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            enable_q  <= enable;
            enable_qq <= enable_q;
            valid_out <= enable_qq;
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
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,
    input  q_data_t weights[NUM_NEURONS],
    input  acc_t biases[NUM_NEURONS],
    output q_data_t data_out[NUM_NEURONS],
    output logic valid_out
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_out <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            if (all_mac_valid) begin
                if (count_out == INPUT_SIZE-1) begin
                    count_out <= 0;
                    valid_out <= 1'b1;
                end else begin
                    count_out <= count_out + 1'b1;
                end
            end
        end
    end

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

BINARYCLASS_FC_OUT_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_out #(
    parameter int NUM_NEURONS = 2,
    parameter signed LAYER_SCALE = 5,
    parameter signed BIAS_SCALE  = 1
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in[NUM_NEURONS],
    input  q_data_t weights[NUM_NEURONS],
    input  acc_t bias,
    output q_data_t data_out,
    output logic valid_out
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
            assign mult_res[i] = $signed(data_in[i]) * $signed(weights[i]);
        end
    endgenerate

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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_pipe2 <= '0;
            valid_p2  <= 1'b0;
        end else begin
            sum_pipe2 <= sum_stage2;
            valid_p2  <= valid_p1;
        end
    end

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

BINARYCLASS_RELU_LAYER_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module relu_layer (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,
    output q_data_t data_out,
    output logic valid_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                data_out <= (data_in < 0) ? 8'd0 : data_in;
        end
    end

endmodule
'''

BINARYCLASS_SIGMOID_LAYER_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module sigmoid_layer (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  q_data_t data_in,
    output q_data_t data_out,
    output logic valid_out
);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                if (data_in >= 0)
                    data_out <= SIGMOID_MAX;
                else
                    data_out <= SIGMOID_MIN;
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


def write_embedded_rtl_templates(sv_dir: Path, weight_width: int, *, write_submodules: bool = True, has_softmax: bool = False) -> None:
    """Write embedded quant_pkg and optionally mac, fc_in, fc_out, relu_layer to sv_dir.
    has_softmax is accepted but ignored (no softmax generation in minimal mapper)."""
    sv_dir.mkdir(parents=True, exist_ok=True)

    (sv_dir / "quant_pkg.sv").write_text(_get_quant_pkg_content(weight_width), encoding="utf-8")
    if write_submodules:
        (sv_dir / "mac.sv").write_text(_pyramidtech_wrap(EMBEDDED_MAC_SV, "mac.sv", "Multiply-accumulate unit for FC layers."), encoding="utf-8")
        (sv_dir / "fc_in.sv").write_text(_pyramidtech_wrap(EMBEDDED_FC_IN_SV, "fc_in.sv", "FC input compute block."), encoding="utf-8")
        (sv_dir / "fc_out.sv").write_text(_pyramidtech_wrap(EMBEDDED_FC_OUT_SV, "fc_out.sv", "FC output compute block."), encoding="utf-8")
        (sv_dir / "relu_layer.sv").write_text(_pyramidtech_wrap(EMBEDDED_RELU_LAYER_SV, "relu_layer.sv", "ReLU activation layer (scalar)."), encoding="utf-8")
        (sv_dir / "relu_layer_array.sv").write_text(_pyramidtech_wrap(EMBEDDED_RELU_LAYER_ARRAY_SV, "relu_layer_array.sv", "ReLU activation layer (array, for legacy)."), encoding="utf-8")
        LOGGER.info("Wrote embedded RTL templates (quant_pkg, mac, fc_in, fc_out, relu_layer, relu_layer_array)")
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
) -> List[Path]:
    """Generate ROM modules with fc_N_proj_in/out naming for parameterized hierarchical flow."""
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
        mem_in_w = f"mem_files/{prefix}_in_weights.mem"
        mem_in_b = f"mem_files/{prefix}_in_bias.mem"

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
        mem_out_w = f"mem_files/{prefix}_out_weights.mem"
        mem_out_b = f"mem_files/{prefix}_out_bias.mem"

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
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in_reg),
        .data_in(data_in_reg),
        .weights(weights_rom_data),
        .biases(bias_rom_data),
        .data_out(data_out),
        .valid_out(valid_out)
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
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_pipeline_reg),
        .data_in(data_in_reg),
        .weights(weights_rom_data_reg),
        .bias(bias_rom_data_reg),
        .data_out(data_out),
        .valid_out(valid_out_engine)
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
) -> Path:
    """Generate top module with streaming control logic. has_softmax is accepted but ignored."""
    active_layers = [l for l in layers if l.layer_type != "flatten"]
    linear_layers = [l for l in active_layers if l.layer_type == "linear"]

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
                    instantiations.extend([
                        f"    fc_in_layer #(",
                        f"        .NUM_NEURONS({num_neurons}),",
                        f"        .INPUT_SIZE({in_features}),",
                        f"        .BIAS_SCALE(0),",
                        f"        .LAYER_SCALE(12)",
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
                    connections.append(f"    // fc_out_layer for {fc_name}")
                    connections.append("")
                    instantiations.extend([
                        f"    fc_out_layer #(",
                        f"        .NUM_NEURONS({in_features}),",
                        f"        .ROM_DEPTH({num_neurons}),",
                        f"        .LAYER_SCALE(5),",
                        f"        .BIAS_SCALE(1)",
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
    frac_bits: int
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
    layers: List[LayerInfo]
) -> Path:
    """Generate netlist.json."""
    netlist = {
        "model_name": model_name,
        "layers": []
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
# Binaryclass_nn Format (AXI4-Stream, binaryclass_NN module name, reference structure)
# -----------------------------------------------------------------------------

def emit_binaryclass_nn_format(
    out_dir: Path,
    layers: List[LayerInfo],
    input_size: int,
    weight_width: int,
    scale: int,
) -> None:
    """Emit RTL in binaryclass_nn format: binaryclass_NN module, AXI4-Stream ports, reference structure.
    Requires exactly 3 FC layers (fc1, fc2, fc3)."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    if len(linear_layers) != 3:
        raise RuntimeError(
            f"binaryclass_nn format requires exactly 3 FC layers; found {len(linear_layers)}. "
            "Use hierarchical or flattened format for other topologies."
        )
    for i, fc in enumerate(linear_layers):
        if fc.name != f"fc{i + 1}":
            raise RuntimeError(f"binaryclass_nn format expects fc1, fc2, fc3; got {fc.name}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = out_dir / "mem_files"
    mem_dir.mkdir(parents=True, exist_ok=True)

    # Layer dimensions
    fc1, fc2, fc3 = linear_layers[0], linear_layers[1], linear_layers[2]
    fc1_in, fc1_out = fc1.in_features or 0, fc1.out_features or 0
    fc2_in, fc2_out = fc2.in_features or 0, fc2.out_features or 0
    fc3_in, fc3_out = fc3.in_features or 0, fc3.out_features or 0
    fc1_rom_depth = fc2_in  # FC1_ROM_DEPTH = FC2_INPUT_SIZE
    fc2_rom_depth = fc3_in  # FC2_ROM_DEPTH = FC3_INPUT_SIZE
    fc3_rom_depth = 1

    # 1. Write embedded binaryclass_nn modules
    (out_dir / "quant_pkg.sv").write_text(
        f"`define Q_INT{weight_width}\n" + BINARYCLASS_QUANT_PKG_SV, encoding="utf-8"
    )
    (out_dir / "mac.sv").write_text(BINARYCLASS_MAC_SV, encoding="utf-8")
    (out_dir / "fc_in.sv").write_text(BINARYCLASS_FC_IN_SV, encoding="utf-8")
    (out_dir / "fc_out.sv").write_text(BINARYCLASS_FC_OUT_SV, encoding="utf-8")
    (out_dir / "relu_layer.sv").write_text(BINARYCLASS_RELU_LAYER_SV, encoding="utf-8")
    (out_dir / "sigmoid_layer.sv").write_text(BINARYCLASS_SIGMOID_LAYER_SV, encoding="utf-8")
    LOGGER.info("  Wrote binaryclass_nn modules (quant_pkg, mac, fc_in, fc_out, relu_layer, sigmoid_layer)")

    # 2. Generate .mem files with binaryclass_nn naming
    # fc_proj_in: fc1 weights (in_f x out_f), fc_proj_out: fc1->fc2 weights (fc2_in x fc1_out)
    # fc_2_proj_in: fc2 weights, fc_2_proj_out: fc2->fc3 weights
    # fc_3_proj_in: fc3 weights, fc_3_proj_out: fc3 output (1x1)
    for layer, (proj_name, in_f, out_f, next_out) in zip(linear_layers, [
        ("fc_proj", fc1_in, fc1_out, fc2_in),
        ("fc_2_proj", fc2_in, fc2_out, fc3_in),
        ("fc_3_proj", fc3_in, fc3_out, 1),
    ]):
        w_np = layer.weight.detach().cpu().numpy().astype(np.float32)
        b_np = layer.bias.detach().cpu().numpy().astype(np.float32) if layer.bias is not None else np.zeros((out_f,), dtype=np.float32)
        wq = float_to_int(w_np, scale, weight_width)
        bq = float_to_int(b_np, scale, weight_width)
        # fc_X_proj_in: weights in_f x out_f
        generate_quant_pkg_style_weight_mem(wq, mem_dir / f"{proj_name}_in_weights.mem", layer.name, in_f, out_f, weight_width)
        generate_quant_pkg_style_bias_mem(bq, mem_dir / f"{proj_name}_in_biases.mem", out_f, weight_width)
        # fc_X_proj_out: weights next_out x out_f (ROM has next_out rows, each row = out_f weights)
        # fc_X_proj_out: ROM has next_out rows, each row has out_f weights. Matrix next_out x out_f.
        # For fc1_out: fc2_in x fc1_out. Our fc2 has in=fc2_in, out=fc2_out. The fc1_out projects fc1->fc2.
        # We need weights (next_out, out_f). Use identity-like when next_out==out_f, else zeros.
        if next_out == out_f:
            w_out = np.eye(next_out, out_f, dtype=np.float32) * (scale / 256.0)
            w_out = float_to_int(w_out, scale, weight_width)
        else:
            w_out = float_to_int(np.zeros((next_out, out_f), dtype=np.float32), scale, weight_width)
        generate_quant_pkg_style_weight_mem(w_out, mem_dir / f"{proj_name}_out_weights.mem", layer.name, out_f, next_out, weight_width)
        bq_out = bq if next_out == 1 else float_to_int(np.zeros((next_out,), dtype=np.float32), scale, weight_width)
        generate_quant_pkg_style_bias_mem(bq_out, mem_dir / f"{proj_name}_out_biases.mem", next_out, weight_width)

    # 3. Generate ROMs with binaryclass_nn naming
    for layer, (proj_name, in_f, out_f, next_out) in zip(linear_layers, [
        ("fc_proj", fc1_in, fc1_out, fc2_in),
        ("fc_2_proj", fc2_in, fc2_out, fc3_in),
        ("fc_3_proj", fc3_in, fc3_out, 1),
    ]):
        _generate_binaryclass_nn_rom(out_dir, f"{proj_name}_in", in_f, out_f, weight_width, mem_dir, "weights")
        _generate_binaryclass_nn_rom(out_dir, f"{proj_name}_in_bias", 1, out_f, weight_width, mem_dir, "biases")
        rom_d = fc2_in if proj_name == "fc_proj" else fc3_in if proj_name == "fc_2_proj" else 1
        _generate_binaryclass_nn_rom(out_dir, f"{proj_name}_out", rom_d, out_f, weight_width, mem_dir, "weights")
        _generate_binaryclass_nn_rom(out_dir, f"{proj_name}_out_bias", rom_d, 1, weight_width, mem_dir, "biases")

    # 4. Generate fc_in_layer and fc_out_layer (with generate blocks for our sizes)
    _generate_fc_in_layer_binaryclass(out_dir, fc1_in, fc2_in, fc3_in, weight_width, mem_dir)
    _generate_fc_out_layer_binaryclass(out_dir, fc1_rom_depth, fc2_rom_depth, fc3_rom_depth, weight_width, mem_dir)

    # 5. Generate binaryclass_NN and binaryclass_NN_wrapper
    _generate_binaryclass_NN(out_dir, fc1_out, fc2_out, fc3_out, fc1_in, fc2_in, fc3_in, fc1_rom_depth, fc2_rom_depth, fc3_rom_depth)
    _generate_binaryclass_NN_wrapper(out_dir, fc1_out, fc2_out, fc3_out, fc1_in, fc2_in, fc3_in, fc1_rom_depth, fc2_rom_depth, fc3_rom_depth)


def _generate_binaryclass_nn_rom(
    out_dir: Path,
    rom_name: str,
    depth: int,
    width_neurons: int,
    weight_width: int,
    mem_dir: Path,
    mem_type: str,
) -> None:
    """Generate a ROM module with binaryclass_nn naming."""
    base = rom_name.replace("_bias", "")
    mem_file = f"mem_files/{base}_{mem_type}.mem"
    if "bias" in rom_name:
        mem_file = mem_file.replace("_weights", "_biases")
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


def _generate_fc_in_layer_binaryclass(out_dir: Path, l1_in: int, l2_in: int, l3_in: int, weight_width: int, mem_dir: Path) -> None:
    """Generate fc_in_layer with generate blocks for model sizes."""
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
        if (INPUT_SIZE == {l1_in}) begin
            fc_proj_in_weights_rom #(.DEPTH(INPUT_SIZE), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(weight_addr), .data_o(weight_rom_row));
            fc_proj_in_bias_rom #(.DEPTH(1), .WIDTH(NUM_NEURONS*Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(1'b0), .data_o(bias_rom_row));
        end else if (INPUT_SIZE == {l2_in}) begin
            fc_2_proj_in_weights_rom #(.DEPTH(INPUT_SIZE), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(weight_addr), .data_o(weight_rom_row));
            fc_2_proj_in_bias_rom #(.DEPTH(1), .WIDTH(NUM_NEURONS*Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(1'b0), .data_o(bias_rom_row));
        end else if (INPUT_SIZE == {l3_in}) begin
            fc_3_proj_in_weights_rom #(.DEPTH(INPUT_SIZE), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(weight_addr), .data_o(weight_rom_row));
            fc_3_proj_in_bias_rom #(.DEPTH(1), .WIDTH(NUM_NEURONS*Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(1'b0), .data_o(bias_rom_row));
        end
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
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in_reg),
        .data_in(data_in_reg),
        .weights(weights_rom_data),
        .biases(bias_rom_data),
        .data_out(data_out),
        .valid_out(valid_out)
    );

endmodule
"""
    (out_dir / "fc_in_layer.sv").write_text(_pyramidtech_wrap(body, "fc_in_layer.sv", "FC input layer with ROM"), encoding="utf-8")


def _generate_fc_out_layer_binaryclass(out_dir: Path, rom1: int, rom2: int, rom3: int, weight_width: int, mem_dir: Path) -> None:
    """Generate fc_out_layer with generate blocks for ROM depths."""
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
    q_data_t data_in_reg[NUM_NEURONS];

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
        if (ROM_DEPTH == {rom1}) begin
            fc_proj_out_weights_rom #(.DEPTH(ROM_DEPTH), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(weights_rom_addr), .data_o(weights_rom_data_raw));
            fc_proj_out_bias_rom #(.DEPTH(ROM_DEPTH), .WIDTH(Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(bias_rom_addr), .data_o(bias_rom_data));
        end else if (ROM_DEPTH == {rom2}) begin
            fc_2_proj_out_weights_rom #(.DEPTH(ROM_DEPTH), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(weights_rom_addr), .data_o(weights_rom_data_raw));
            fc_2_proj_out_bias_rom #(.DEPTH(ROM_DEPTH), .WIDTH(Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(bias_rom_addr), .data_o(bias_rom_data));
        end else if (ROM_DEPTH == {rom3}) begin
            fc_3_proj_out_weights_rom #(.DEPTH(ROM_DEPTH), .WIDTH(NUM_NEURONS*Q_WIDTH)) weights_rom_inst (
                .clk_i(clk), .addr_i(1'b0), .data_o(weights_rom_data_raw));
            fc_3_proj_out_bias_rom #(.DEPTH(ROM_DEPTH), .WIDTH(Q_WIDTH*4)) bias_rom_inst (
                .clk_i(clk), .addr_i(1'b0), .data_o(bias_rom_data));
        end
    endgenerate

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : SPLIT_WEIGHTS
            assign weights_rom_data[i] = weights_rom_data_raw[i*Q_WIDTH +: Q_WIDTH];
        end
    endgenerate

    integer k;
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
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_pipeline_reg),
        .data_in(data_in_reg),
        .weights(weights_rom_data_reg),
        .bias(bias_rom_data_reg),
        .data_out(data_out),
        .valid_out(valid_out_engine)
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
        files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "fc_in_layer.sv", "fc_out_layer.sv", "relu_layer.sv", "sigmoid_layer.sv"]
        files.extend(["fc_proj_in_weights_rom.sv", "fc_proj_in_bias_rom.sv", "fc_proj_out_weights_rom.sv", "fc_proj_out_bias_rom.sv"])
        files.extend(["fc_2_proj_in_weights_rom.sv", "fc_2_proj_in_bias_rom.sv", "fc_2_proj_out_weights_rom.sv", "fc_2_proj_out_bias_rom.sv"])
        files.extend(["fc_3_proj_in_weights_rom.sv", "fc_3_proj_in_bias_rom.sv", "fc_3_proj_out_weights_rom.sv", "fc_3_proj_out_bias_rom.sv"])
        files.append("binaryclass_NN.sv")
        top_file = "binaryclass_NN.sv"
    elif parameterized_layers:
        files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "relu_layer.sv", "relu_layer_array.sv", "fc_in_layer.sv", "fc_out_layer.sv"]
        for idx in range(len(linear_layers)):
            prefix = _proj_prefix(idx)
            files.append(f"{prefix}_in_weights_rom.sv")
            files.append(f"{prefix}_in_bias_rom.sv")
            if idx + 1 < len(linear_layers):
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