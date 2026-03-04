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
# PyramidTech / binaryclass_nn format constants
# -----------------------------------------------------------------------------
PYRAMIDTECH_HEADER = r'''/****************************************************************************************
"PYRAMIDTECH CONFIDENTIAL

Copyright (c) 2026 PyramidTech LLC. All rights reserved.

This file contains proprietary and confidential information of PyramidTech LLC.
The information contained herein is unpublished and subject to trade secret
protection. No part of this file may be reproduced, modified, distributed,
transmitted, disclosed, or used in any form or by any means without the
prior written permission of PyramidTech LLC.

This material must be returned immediately upon request by PyramidTech LLC"
/****************************************************************************************
'''


def _pyramidtech_wrap(content: str, file_name: str, description: str) -> str:
    """Wrap RTL content with PyramidTech header and keywords."""
    header = (
        PYRAMIDTECH_HEADER
        + f"File name:      {file_name}\n  \nDescription:    {description}\n  \nAuthor:         Ahmed Abou-Auf\n  \nChange History:\n02-25-2026     AA  Initial Release\n  \n****************************************************************************************/\n\n"
    )
    return header + '`begin_keywords "1800-2012"\n' + content.rstrip() + "\n`end_keywords\n"


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
# Embedded RTL Templates
# -----------------------------------------------------------------------------

def _get_quant_pkg_content(weight_width: int) -> str:
    """Generate quant_pkg.sv in binaryclass_nn format with Q_WIDTH based on weight_width."""
    if weight_width not in (4, 8, 16):
        weight_width = 16
    define_line = f"`define Q_INT{weight_width}\n"
    default_def = "`ifndef Q_INT4\n`ifndef Q_INT8\n`ifndef Q_INT16\n`define Q_INT16\n`endif\n`endif\n`endif\n"
    body = define_line + default_def + '''package quant_pkg;

  timeunit 1ns;
  timeprecision 1ps;

  `ifdef Q_INT4
    localparam int Q_WIDTH = 32'd4;
  `elsif Q_INT8
    localparam int Q_WIDTH = 32'd8;
  `elsif Q_INT16
    localparam int Q_WIDTH = 32'd16;
  `else
    localparam int Q_WIDTH = 32'd8;
  `endif

  typedef logic signed [Q_WIDTH-1:0]       q_data_t;
  typedef logic signed [2*Q_WIDTH-1:0]     q_mult_t;
  typedef logic signed [4*Q_WIDTH-1:0]     acc_t;

  localparam int DATA_WIDTH = 32'd32;
  localparam int KEEP_WIDTH = 32'd4;

  localparam q_data_t Q_MAX = {1'b0, {Q_WIDTH-1{1'b1}}};
  localparam q_data_t Q_MIN = {1'b1, {Q_WIDTH-1{1'b0}}};

  localparam acc_t ACC_Q_MAX = acc_t'(Q_MAX);
  localparam acc_t ACC_Q_MIN = acc_t'(Q_MIN);

  localparam acc_t ACC_FULL_MAX = {1'b0, {4*Q_WIDTH-1{1'b1}}};
  localparam acc_t ACC_FULL_MIN = {1'b1, {4*Q_WIDTH-1{1'b0}}};

  `ifdef Q_INT4
    localparam int FRAC_WIDTH = 2;
  `elsif Q_INT8
    localparam int FRAC_WIDTH = 4;
  `elsif Q_INT16
    localparam int FRAC_WIDTH = 8;
  `endif

  `ifdef Q_INT4
    localparam int SATURATION_H = 32'sd7;
  `elsif Q_INT8
    localparam int SATURATION_H = 32'sd127;
  `elsif Q_INT16
    localparam int SATURATION_H = 32'sd32767;
  `endif

  `ifdef Q_INT4
    localparam int SATURATION_L = -32'sd8;
  `elsif Q_INT8
    localparam int SATURATION_L = -32'sd128;
  `elsif Q_INT16
    localparam int SATURATION_L = -32'sd32768;
  `endif

  `ifdef Q_INT4
    localparam int LIMIT_H = 4'sd7;
  `elsif Q_INT8
    localparam int LIMIT_H = 8'sd127;
  `elsif Q_INT16
    localparam int LIMIT_H = 16'sd32767;
  `endif

  `ifdef Q_INT4
    localparam int LIMIT_L = -4'sd8;
  `elsif Q_INT8
    localparam int LIMIT_L = -8'sd128;
  `elsif Q_INT16
    localparam int LIMIT_L = -16'sd32768;
  `endif

  localparam q_data_t SIGMOID_MAX = 1 << (Q_WIDTH - 2);
  localparam q_data_t SIGMOID_MIN = 8'h0;

endpackage: quant_pkg
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

   output logic out_valid,
   output acc_t acc
);

   (* use_dsp = "yes" *) logic signed [63:0] mult_reg;
   logic signed [63:0] acc_reg;
   logic               enable_q;
   logic               enable_qq;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         mult_reg <= '0;
      else if (enable)
         mult_reg <= $signed(a) * $signed(b);
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
         acc_reg <= '0;
      else if (clear)
         acc_reg <= '0;
      else if (enable)
         acc_reg <= acc_reg + mult_reg;
   end

   always_ff @(posedge clk) begin
      if (acc_reg > $signed(32'h7FFFFFFF))
         acc <= 32'h7FFFFFFF;
      else if (acc_reg < $signed(32'h80000000))
         acc <= 32'h80000000;
      else
         acc <= acc_reg[31:0];
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         out_valid <= 1'b0;
      end
      else if (clear)begin
         enable_q  <= 1'b0;
         enable_qq <= 1'b0;
         out_valid <= 1'b0;
      end
      else if (enable)begin
         enable_q  <= enable;
         enable_qq <= enable_q;
         out_valid <= enable_qq;
      end
   end

endmodule
'''


EMBEDDED_FC_IN_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_in #(
   parameter int NUM_NEURONS = 8,
   parameter int INPUT_SIZE  = 16
)(
   input  logic                 clk,
   input  logic                 rst_n,

   input  logic                 in_valid,
   input  q_data_t              data_in,

   input  q_data_t              weights [NUM_NEURONS],
   input  q_data_t              bias    [NUM_NEURONS],

   output q_data_t              fc_out  [NUM_NEURONS],
   output logic                 out_valid
);

   acc_t mac_acc     [NUM_NEURONS];
   acc_t mac_acc_tmp [NUM_NEURONS];
   acc_t acc_tmp [NUM_NEURONS];
   acc_t bias_32 [NUM_NEURONS];

   logic [$clog2(INPUT_SIZE+1)-1:0] out_count;
   logic mac_enable;
   logic mac_clear;
   logic [NUM_NEURONS-1:0]mac_out_valid;
   logic all_mac_valid;

   genvar i;
   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_MACS
         mac mac_inst (
            .clk       (clk),
            .rst_n     (rst_n),
            .enable    (mac_enable),
            .clear     (mac_clear),
            .a         (data_in),
            .b         (weights[i]),
            .out_valid (mac_out_valid[i]),
            .acc       (mac_acc[i])
         );
      end
   endgenerate

   assign all_mac_valid = &mac_out_valid;

   always_ff @(posedge clk) begin
      if (!rst_n) begin
         out_count  <= '0;
         out_valid <= 1'b0;
      end else begin
         out_valid <= 1'b0;
         if (all_mac_valid) begin
            if (out_count == INPUT_SIZE-1) begin
               out_count  <= '0;
               out_valid <= 1'b1;
            end else begin
               out_count <= out_count + 1'b1;
            end
         end
      end
   end

   assign mac_enable = in_valid;
   assign mac_clear  = out_valid;

   generate
      for (i = 0; i < NUM_NEURONS; i++) begin : FC_BIAS_ADD
         assign mac_acc_tmp[i] = (mac_acc[i] >>> FRAC_WIDTH);
         assign bias_32[i] = {{32-Q_WIDTH{bias[i][Q_WIDTH-1]}}, bias[i]};
         assign acc_tmp[i] = mac_acc_tmp[i] + bias_32[i];
         always_ff @(posedge clk) begin
            if (!rst_n)
               fc_out[i] <= '0;
            else if (out_count == INPUT_SIZE-1)
               if (acc_tmp[i] > SATURATION_H)
                  fc_out[i] <= LIMIT_H;
               else if (acc_tmp[i] < SATURATION_L)
                  fc_out[i] <= $signed(LIMIT_L);
               else
                  fc_out[i] <= acc_tmp[i][Q_WIDTH:0];
         end
      end
   endgenerate

endmodule
'''


EMBEDDED_FC_OUT_SV = '''`timescale 1ns/1ps
import quant_pkg::*;

module fc_out #(
    parameter int NUM_MULT = 2
)(
    input  logic clk,
    input  logic rst_n,
    input  logic en,

    input  q_data_t    fc_in   [NUM_MULT],
    input  q_data_t    weights [NUM_MULT],
    input  q_data_t    bias,

    output logic       fc_out_valid,
    output q_data_t    fc_out
);

    (* use_dsp = "yes" *) logic signed [63:0] mult_res [NUM_MULT];
    logic signed [63:0] mult_res_tmp [NUM_MULT];
    logic signed [63:0] sum_comb;

    genvar i;
    generate
        for (i = 0; i < NUM_MULT; i++) begin : GEN_MULT
            assign mult_res[i]     = $signed(fc_in[i]) * $signed(weights[i]);
            assign mult_res_tmp[i] = (mult_res[i] >>> FRAC_WIDTH);
        end
    endgenerate

    integer k;
    always_comb begin
        sum_comb = {{64-Q_WIDTH{bias[Q_WIDTH-1]}}, bias};
        for (k = 0; k < NUM_MULT; k++)
            sum_comb = sum_comb + mult_res_tmp[k];
    end

    always_ff @(posedge clk or negedge rst_n) begin
       if (!rst_n) begin
           fc_out       <= '0;
           fc_out_valid <= 1'b0;
       end
       else if (en) begin
          fc_out_valid <= 1'b1;
          if (sum_comb > SATURATION_H)
             fc_out <= LIMIT_H;
          else if (sum_comb < SATURATION_L)
             fc_out <= $signed(LIMIT_L);
          else
             fc_out <= sum_comb[Q_WIDTH:0];
       end
       else
          fc_out_valid <= 1'b0;
    end

endmodule
'''


EMBEDDED_RELU_LAYER_SV = '''module relu_layer 
  import quant_pkg::*;
#(
  parameter int NUM_NEURONS = 8
)(
  input  logic        clk_i,
  input  logic        rst_n_i,
  input  logic        valid_i,
  input  q_data_t     data_i [NUM_NEURONS],
  output q_data_t     data_o [NUM_NEURONS],
  output logic        valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  logic valid_pipeline_q;
  integer i;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : relu_pipeline
    if (!rst_n_i)
      valid_pipeline_q <= 1'b0;
    else
      valid_pipeline_q <= valid_i;
  end : relu_pipeline

  assign valid_o = valid_pipeline_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : relu_data
    if (!rst_n_i) begin
      for (i = 0; i < NUM_NEURONS; i++)
        data_o[i] <= '0;
    end
    else if (valid_i) begin
      for (i = 0; i < NUM_NEURONS; i++) begin
        if (data_i[i] < 0)
          data_o[i] <= '0;
        else
          data_o[i] <= data_i[i];
      end
    end
  end : relu_data

endmodule : relu_layer
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
        (sv_dir / "relu_layer.sv").write_text(_pyramidtech_wrap(EMBEDDED_RELU_LAYER_SV, "relu_layer.sv", "ReLU activation layer."), encoding="utf-8")
        LOGGER.info("Wrote embedded RTL templates (quant_pkg, mac, fc_in, fc_out, relu_layer)")
    else:
        LOGGER.info("Wrote quant_pkg.sv only (flattened mode)")


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

    body = f"""module weight_rom_{layer_name} #(
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

endmodule : weight_rom_{layer_name}
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

    body = f"""module bias_rom_{layer_name} #(
  parameter int WIDTH = {packed_width}
) (
  input  logic        clk_i,
  output logic [WIDTH-1:0] data_o
);

  timeunit 1ns;
  timeprecision 1ps;

  (* rom_style = "block" *)
  logic [WIDTH-1:0] mem [0:0];

  initial begin
    $readmemh("{mem_file}", mem);
  end

  always_ff @(posedge clk_i) begin : read_port
    data_o <= mem[0];
  end : read_port

endmodule : bias_rom_{layer_name}
"""
    content = _pyramidtech_wrap(body, f"bias_rom_{layer_name}.sv", f"ROM of biases for {layer_name} layer")
    out_path = out_dir / f"bias_rom_{layer_name}.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def generate_fc_layer_wrapper(
    layer_name: str,
    in_features: int,
    num_neurons: int,
    data_width: int,
    weight_width: int,
    out_dir: Path
) -> Path:
    """Generate fc_layer_<layer>.sv wrapper module in binaryclass_nn format."""
    body = f"""module fc_layer_{layer_name} 
  import quant_pkg::*;
#(
  parameter int NUM_NEURONS = {num_neurons},
  parameter int INPUT_SIZE  = {in_features}
)(
  input  logic        clk_i,
  input  logic        rst_n_i,
  input  logic        valid_i,
  input  q_data_t     data_i,
  output q_data_t     data_o [NUM_NEURONS],
  output logic        valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  logic [$clog2(INPUT_SIZE)-1:0] weight_addr_q;
  logic [NUM_NEURONS*Q_WIDTH-1:0] weight_row_s;
  logic [NUM_NEURONS*Q_WIDTH-1:0] bias_row_s;

  q_data_t weights_s [NUM_NEURONS];
  q_data_t bias_s    [NUM_NEURONS];

  logic    valid_i_q;
  q_data_t data_i_q;

  genvar i;

  weight_rom_{layer_name} u_weight_rom (
    .clk_i  (clk_i),
    .addr_i (weight_addr_q),
    .data_o (weight_row_s)
  );

  bias_rom_{layer_name} u_bias_rom (
    .clk_i  (clk_i),
    .data_o (bias_row_s)
  );

  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : WEIGHT_SLICE
      assign weights_s[i] = weight_row_s[i*Q_WIDTH +: Q_WIDTH];
    end
  endgenerate

  generate
    for (i = 0; i < NUM_NEURONS; i++) begin : BIAS_SLICE
      assign bias_s[i] = bias_row_s[i*Q_WIDTH +: Q_WIDTH];
    end
  endgenerate

  always_ff @(posedge clk_i or negedge rst_n_i) begin : input_regs
    if (!rst_n_i) begin
      valid_i_q <= 1'b0;
      data_i_q  <= '0;
    end
    else begin
      valid_i_q <= valid_i;
      data_i_q  <= data_i;
    end
  end : input_regs

  always_ff @(posedge clk_i or negedge rst_n_i) begin : addr_counter
    if (!rst_n_i)
      weight_addr_q <= '0;
    else if (valid_i) begin
      if (weight_addr_q == INPUT_SIZE-1)
        weight_addr_q <= '0;
      else
        weight_addr_q <= weight_addr_q + 1'b1;
    end
  end : addr_counter

  fc_in #(
    .NUM_NEURONS (NUM_NEURONS),
    .INPUT_SIZE  (INPUT_SIZE)
  ) u_fc_in (
    .clk       (clk_i),
    .rst_n     (rst_n_i),
    .in_valid  (valid_i_q),
    .data_in   (data_i_q),
    .weights   (weights_s),
    .bias      (bias_s),
    .fc_out    (data_o),
    .out_valid (valid_o)
  );

endmodule : fc_layer_{layer_name}
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
    data_o_port = "output q_data_t     data_o" if num_neurons == 1 else "output q_data_t     data_o [NUM_NEURONS]"
    body = f"""module fc_out_layer_{layer_name} 
  import quant_pkg::*;
#(
  parameter int NUM_NEURONS = {num_neurons},
  parameter int NUM_INPUTS  = {num_inputs}
)(
  input  logic        clk_i,
  input  logic        rst_n_i,
  input  logic        valid_i,
  input  q_data_t     data_i [NUM_INPUTS],
  {data_o_port},
  output logic        valid_o
);

  timeunit 1ns;
  timeprecision 1ps;

  logic [$clog2(NUM_NEURONS)-1:0] addr_cnt_q;
  logic valid_in_reg_q;
  q_data_t data_i_reg_q [NUM_INPUTS];

  logic [NUM_INPUTS*Q_WIDTH-1:0] weights_rom_data_raw_s;
  q_data_t weights_rom_data_s [NUM_INPUTS];
  q_data_t bias_rom_data_s [NUM_NEURONS];
  logic [NUM_NEURONS*Q_WIDTH-1:0] bias_rom_data_raw_s;
  q_data_t bias_rom_data_reg_q;

  logic fc_out_valid_s;
  q_data_t fc_out_tmp_s;
  logic [$clog2(NUM_NEURONS+1)-1:0] fc_out_cnt_q;

  always_ff @(posedge clk_i or negedge rst_n_i) begin : addr_counter
    if (!rst_n_i)
      addr_cnt_q <= '0;
    else if (addr_cnt_q == NUM_NEURONS-1)
      addr_cnt_q <= '0;
    else if (valid_i || addr_cnt_q > 0)
      addr_cnt_q <= addr_cnt_q + 1'b1;
  end : addr_counter

  always_ff @(posedge clk_i or negedge rst_n_i) begin : input_regs
    if (!rst_n_i) begin
      valid_in_reg_q <= 1'b0;
      for (int j = 0; j < NUM_INPUTS; j++)
        data_i_reg_q[j] <= '0;
    end
    else begin
      if (addr_cnt_q == '0) begin
        valid_in_reg_q <= valid_i;
        data_i_reg_q   <= data_i;
      end
      else if (addr_cnt_q == NUM_NEURONS) begin
        valid_in_reg_q <= 1'b0;
        for (int j = 0; j < NUM_INPUTS; j++)
          data_i_reg_q[j] <= '0;
      end
    end
  end : input_regs

  logic valid_pipeline_q2;
  logic [$clog2(NUM_NEURONS)-1:0] addr_cnt_q_d;

  weight_rom_{layer_name} u_weights_rom (
    .clk_i  (clk_i),
    .addr_i (addr_cnt_q),
    .data_o (weights_rom_data_raw_s)
  );

  bias_rom_{layer_name} u_bias_rom (
    .clk_i  (clk_i),
    .data_o (bias_rom_data_raw_s)
  );

  always_ff @(posedge clk_i or negedge rst_n_i) begin : rom_align_reg
    if (!rst_n_i) begin
      valid_pipeline_q2 <= 1'b0;
      addr_cnt_q_d      <= '0;
    end
    else begin
      valid_pipeline_q2 <= valid_in_reg_q;
      addr_cnt_q_d      <= addr_cnt_q;
    end
  end : rom_align_reg

  genvar i;
  generate
    for (i = 0; i < NUM_INPUTS; i++) begin : INPUT_SPLIT
      assign weights_rom_data_s[i] = weights_rom_data_raw_s[i*Q_WIDTH +: Q_WIDTH];
    end
  endgenerate

  genvar k;
  generate
    for (k = 0; k < NUM_NEURONS; k++) begin : SPLIT_BIAS
      assign bias_rom_data_s[k] = bias_rom_data_raw_s[k*Q_WIDTH +: Q_WIDTH];
    end
  endgenerate

  always_ff @(posedge clk_i or negedge rst_n_i) begin : bias_reg
    if (!rst_n_i)
      bias_rom_data_reg_q <= '0;
    else
      bias_rom_data_reg_q <= bias_rom_data_s[addr_cnt_q_d];
  end : bias_reg

  fc_out #(.NUM_MULT(NUM_INPUTS)) u_fc_out (
    .clk         (clk_i),
    .rst_n       (rst_n_i),
    .en          (valid_pipeline_q2),
    .fc_in       (data_i_reg_q),
    .weights     (weights_rom_data_s),
    .bias        (bias_rom_data_reg_q),
    .fc_out_valid(fc_out_valid_s),
    .fc_out      (fc_out_tmp_s)
  );

  always_ff @(posedge clk_i or negedge rst_n_i) begin : output_shift
    if (!rst_n_i) begin
      for (int n = 0; n < NUM_NEURONS; n++)
        fc_out_q[n] <= '0;
      fc_out_cnt_q <= '0;
      valid_o      <= 1'b0;
    end
    else begin
      valid_o <= 1'b0;
      if (fc_out_valid_s) begin
        for (int n = 0; n < NUM_NEURONS-1; n++)
          fc_out_q[n] <= fc_out_q[n+1];
        fc_out_q[NUM_NEURONS-1] <= fc_out_tmp_s;
        if (fc_out_cnt_q == NUM_NEURONS-1) begin
          valid_o <= 1'b1;
          fc_out_cnt_q <= '0;
        end
        else if (fc_out_cnt_q < NUM_NEURONS)
          fc_out_cnt_q <= fc_out_cnt_q + 1'b1;
      end
    end
  end : output_shift

  assign data_o = (NUM_NEURONS == 1) ? fc_out_q[0] : fc_out_q;

endmodule : fc_out_layer_{layer_name}
"""
    # Fix assign for scalar vs array output
    if num_neurons == 1:
        body = body.replace("  assign data_o = (NUM_NEURONS == 1) ? fc_out_q[0] : fc_out_q;", "  assign data_o = fc_out_q[0];")
    else:
        body = body.replace("  assign data_o = (NUM_NEURONS == 1) ? fc_out_q[0] : fc_out_q;", "  assign data_o = fc_out_q;")
    # Add fc_out_q declaration (always array for the shift logic)
    body = body.replace("  logic [$clog2(NUM_NEURONS+1)-1:0] fc_out_cnt_q;", "  logic [$clog2(NUM_NEURONS+1)-1:0] fc_out_cnt_q;\n  q_data_t fc_out_q [NUM_NEURONS];")
    content = _pyramidtech_wrap(body, f"fc_out_layer_{layer_name}.sv", f"Fully-connected output layer {layer_name} with sequential ROM access.")
    out_path = out_dir / f"fc_out_layer_{layer_name}.sv"
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
) -> Path:
    """Generate top module with streaming control logic. has_softmax is accepted but ignored."""
    active_layers = [l for l in layers if l.layer_type != "flatten"]

    port_decls = [
        "  input  logic        clk_i,",
        "  input  logic        rst_n_i,",
        "  input  logic        valid_i,",
        "  input  q_data_t     data_i,",
    ]

    signal_decls = []
    instantiations = []
    connections = []

    stream_valid = "valid_i"
    stream_data = "data_i"
    stream_vector = None
    stream_vector_valid = None
    prev_fc_out = None
    prev_fc_valid = None

    for i, layer in enumerate(active_layers):
        if layer.layer_type == "linear":
            fc_name = layer.name
            num_neurons = layer.out_features
            in_features = layer.in_features

            fc_valid_out = f"{fc_name}_valid_s"
            fc_out = f"{fc_name}_out_s"

            signal_decls.extend([
                f"  logic      {fc_valid_out};",
                _q_data_decl(fc_out, num_neurons, "  "),
                ""
            ])

            if fc_name == "fc1":
                instantiations.extend([
                    f"  fc_layer_{fc_name} #(",
                    f"    .NUM_NEURONS  ({num_neurons}),",
                    f"    .INPUT_SIZE   ({in_features})",
                    f"  ) u_{fc_name} (",
                    f"    .clk_i    (clk_i),",
                    f"    .rst_n_i  (rst_n_i),",
                    f"    .valid_i  ({stream_valid}),",
                    f"    .data_i   ({stream_data}),",
                    f"    .data_o   ({fc_out}),",
                    f"    .valid_o  ({fc_valid_out})",
                    f"  );",
                    ""
                ])
            else:
                connections.append(f"  // fc_out_layer_{fc_name} uses {stream_vector} and {stream_vector_valid}")
                connections.append("")
                instantiations.extend([
                    f"  fc_out_layer_{fc_name} #(",
                    f"    .NUM_NEURONS ({num_neurons}),",
                    f"    .NUM_INPUTS  ({in_features})",
                    f"  ) u_{fc_name} (",
                    f"    .clk_i   (clk_i),",
                    f"    .rst_n_i (rst_n_i),",
                    f"    .valid_i ({stream_vector_valid}),",
                    f"    .data_i  ({stream_vector}),",
                    f"    .data_o  ({fc_out}),",
                    f"    .valid_o ({fc_valid_out})",
                    f"  );",
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
            relu_out_vector = f"{relu_name}_out_s"
            relu_out_valid = f"{relu_name}_valid_s"

            signal_decls.extend([
                _q_data_decl(relu_out_vector, num_neurons, "  "),
                f"  logic      {relu_out_valid};",
                ""
            ])

            instantiations.extend([
                f"  relu_layer #(",
                f"    .NUM_NEURONS({num_neurons})",
                f"  ) u_{relu_name} (",
                f"    .clk_i   (clk_i),",
                f"    .rst_n_i (rst_n_i),",
                f"    .valid_i ({relu_in_valid}),",
                f"    .data_i  ({relu_in_vector}),",
                f"    .data_o  ({relu_out_vector}),",
                f"    .valid_o ({relu_out_valid})",
                f"  );",
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

    data_o_line = _q_data_port_decl("data_o", output_size, "output").replace("    ", "  ")
    port_decls.extend([
        "  output logic    valid_o,",
        data_o_line
    ])
    connections.extend([
        f"  assign valid_o  = {last_fc_valid};",
        f"  assign data_o   = {last_fc_out};"
    ])

    body = f"""module {model_name}_top 
  import quant_pkg::*;
(
{chr(10).join(port_decls)}
);

  timeunit 1ns;
  timeprecision 1ps;

{chr(10).join(signal_decls)}

{chr(10).join(connections)}

{chr(10).join(instantiations)}

endmodule : {model_name}_top
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

    out_data_port = _q_data_port_decl("out_data", output_size, "output")
    parts.append(f'''`timescale 1ns/1ps
import quant_pkg::*;

module {{model_name}}_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  q_data_t     in_data,
    output logic        out_valid,
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
        else if (in_valid)
            fc1_weight_addr <= (fc1_weight_addr == FC1_INPUT_SIZE-1) ? '0 : fc1_weight_addr + 1'b1;
    end

    generate
        for (gi = 0; gi < FC1_NEURONS; gi++) begin : FC1_MAC
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_mult_reg[gi] <= '0;
                else if (in_valid) fc1_mult_reg[gi] <= $signed(in_data) * $signed(fc1_weights[gi]);
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) fc1_acc_reg[gi] <= '0;
                else if (fc1_valid_out) fc1_acc_reg[gi] <= '0;
                else if (in_valid) fc1_acc_reg[gi] <= fc1_acc_reg[gi] + fc1_mult_reg[gi];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else if (fc1_valid_out) begin fc1_enable_q <= 1'b0; fc1_enable_qq <= 1'b0; end
        else begin fc1_enable_q <= in_valid; fc1_enable_qq <= fc1_enable_q; end
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
    assign out_valid = fc{i}_valid_out;
    assign out_data = fc{i}_out;
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


def generate_wrapper_module(
    model_name: str,
    layers: List[LayerInfo],
    weight_width: int,
    out_dir: Path,
) -> Path:
    """Generate a thin wrapper module that instantiates the top module."""
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
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_data  (in_data),
        .out_valid(out_valid),
        .out_data (out_data)
    );

endmodule
'''
    content = content.replace("{model_name}", model_name)
    out_path = out_dir / f"{model_name}_wrapper.sv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


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
        .clk_i   (clk),
        .rst_n_i (rst_n),
        .valid_i (in_valid),
        .data_i  (in_data),
        .valid_o (out_valid),
        .data_o  (out_data)
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


def generate_rtl_filelist(
    out_dir: Path,
    model_name: str,
    layers: List[LayerInfo],
) -> Path:
    """Generate rtl_filelist.f in binaryclass_nn format."""
    linear_layers = [l for l in layers if l.layer_type == "linear"]
    root_var = "$ROOT_DIR"
    incdir = f"+incdir+{root_var}/\n"
    lines = [incdir]
    # List all RTL files (commented)
    files = ["quant_pkg.sv", "mac.sv", "fc_in.sv", "fc_out.sv", "relu_layer.sv"]
    for layer in linear_layers:
        files.append(f"weight_rom_{layer.name}.sv")
        files.append(f"bias_rom_{layer.name}.sv")
    for layer in linear_layers:
        if layer.name == "fc1":
            files.append(f"fc_layer_{layer.name}.sv")
        else:
            files.append(f"fc_out_layer_{layer.name}.sv")
    files.append(f"{model_name}_top.sv")
    for f in files:
        lines.append(f"// {root_var}/{f}\n")
    lines.append(f"{root_var}/{model_name}_top.sv\n")
    out_path = out_dir / "rtl_filelist.f"
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path