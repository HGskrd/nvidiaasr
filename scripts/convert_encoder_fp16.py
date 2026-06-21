#!/usr/bin/env python3
"""Dequantize a MatMulNBits (INT4) ONNX model to fp16 (or fp32).

Why: onnx-community ships this model INT4-only. On a weak iGPU the INT4
`MatMulNBits` op pays a per-matmul dequantization cost; a native fp16 model
skips that but moves 4x more weight data. Which wins depends on whether the GPU
is compute- or bandwidth-bound — so this lets us produce an fp16 build and
benchmark it head-to-head against the INT4 one.

What it does:
  1. Rewrites every `com.microsoft::MatMulNBits` node into a plain `MatMul`
     with the weights dequantized into a normal initializer.
  2. Optionally casts the whole graph to fp16 (`--fp16`), keeping the *I/O*
     types as fp32 (`keep_io_types`) so the browser code feeds float32 tensors
     exactly as it does today — no JS changes needed.
  3. Optionally verifies (`--verify`) that the converted model produces outputs
     close to the original on random inputs (CPU EP).

Usage (on the Mac dev machine, no GPU required):
    pip install onnx numpy onnxconverter-common onnxruntime
    python scripts/convert_encoder_fp16.py \
        --in  /path/to/encoder.onnx \
        --out /path/to/encoder.fp16.onnx \
        --fp16 --verify

External data (`encoder.onnx.data`) must sit next to the input .onnx. The
output is written with its own external-data file so it can exceed 2 GB.

NOTE: the MatMulNBits dequant layout (block packing, default zero-point of 8,
B stored transposed as [N, K]) follows the ORT contrib-op spec, but this script
is unverified on your exact export — always run with --verify before trusting
the result, and sanity-check a transcription in the app.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def dequantize_matmulnbits(
    b: np.ndarray,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray],
    k: int,
    n: int,
    bits: int,
    block_size: int,
) -> np.ndarray:
    """Return the dequantized weight as a [K, N] float32 matrix (ready for MatMul)."""
    if bits != 4:
        raise NotImplementedError(f"only 4-bit supported, got {bits}")

    n_blocks = (k + block_size - 1) // block_size
    blob = block_size * bits // 8  # bytes per block (block_size/2 for 4-bit)
    b = b.reshape(n, n_blocks, blob).astype(np.uint8)

    # Two 4-bit values per byte, low nibble first.
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    unpacked = np.empty((n, n_blocks, block_size), dtype=np.float32)
    unpacked[:, :, 0::2] = lo
    unpacked[:, :, 1::2] = hi

    sc = scales.reshape(n, n_blocks, 1).astype(np.float32)

    if zero_points is None:
        zp = np.float32(2 ** (bits - 1))  # default symmetric zero-point = 8
    elif zero_points.dtype == np.uint8 and zero_points.size == n * ((n_blocks + 1) // 2):
        # Packed nibble zero-points: [N, ceil(n_blocks/2)].
        zb = zero_points.reshape(n, (n_blocks + 1) // 2)
        zlo = zb & 0x0F
        zhi = (zb >> 4) & 0x0F
        zfull = np.empty((n, ((n_blocks + 1) // 2) * 2), dtype=np.float32)
        zfull[:, 0::2] = zlo
        zfull[:, 1::2] = zhi
        zp = zfull[:, :n_blocks].reshape(n, n_blocks, 1)
    else:
        zp = zero_points.reshape(n, n_blocks, 1).astype(np.float32)

    deq = (unpacked - zp) * sc                       # [N, n_blocks, block_size]
    deq = deq.reshape(n, n_blocks * block_size)[:, :k]  # [N, K]
    return np.ascontiguousarray(deq.T)               # [K, N] for A[M,K] @ W[K,N]


def convert(model: onnx.ModelProto, fp16: bool) -> int:
    """Replace each MatMulNBits with a dequantized MatMul.

    With fp16=True the matmul runs in fp16 — the weight initializer is stored as
    fp16 and the activation is cast to fp16 on the way in and back to fp32 on the
    way out, so only the heavy matmul is half-precision and every graph interface
    stays fp32 (no JS changes, no fragile whole-graph cast). With fp16=False the
    weights are plain fp32 (un-quantized; ~2x larger again, mainly a sanity test).
    """
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    replaced = 0
    new_nodes = []

    for node in graph.node:
        if not (node.op_type == "MatMulNBits" and node.domain in ("com.microsoft", "")):
            new_nodes.append(node)
            continue

        k = int(_attr(node, "K"))
        n = int(_attr(node, "N"))
        bits = int(_attr(node, "bits", 4))
        block_size = int(_attr(node, "block_size"))

        a_name = node.input[0]
        out_name = node.output[0]
        b = numpy_helper.to_array(inits[node.input[1]])
        scales = numpy_helper.to_array(inits[node.input[2]])
        zero_points = (
            numpy_helper.to_array(inits[node.input[3]])
            if len(node.input) > 3 and node.input[3] and node.input[3] in inits
            else None
        )

        weight = dequantize_matmulnbits(b, scales, zero_points, k, n, bits, block_size)
        w_name = f"{out_name}_dequant_W"

        if fp16:
            graph.initializer.append(numpy_helper.from_array(weight.astype(np.float16), name=w_name))
            a16, y16 = f"{out_name}_a16", f"{out_name}_y16"
            new_nodes.append(helper.make_node("Cast", [a_name], [a16], to=TensorProto.FLOAT16, name=f"{node.name}_castA"))
            new_nodes.append(helper.make_node("MatMul", [a16, w_name], [y16], name=f"{node.name}_mm16"))
            new_nodes.append(helper.make_node("Cast", [y16], [out_name], to=TensorProto.FLOAT, name=f"{node.name}_castY"))
        else:
            graph.initializer.append(numpy_helper.from_array(weight, name=w_name))
            new_nodes.append(helper.make_node("MatMul", [a_name, w_name], [out_name], name=f"{node.name}_dq"))
        replaced += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input encoder.onnx (with .data alongside)")
    ap.add_argument("--out", dest="out", required=True, help="output .onnx path")
    ap.add_argument("--fp16", action="store_true", help="cast graph to fp16 (keeps fp32 I/O)")
    ap.add_argument("--verify", action="store_true", help="compare outputs vs original on random inputs")
    args = ap.parse_args()

    print(f"loading {args.inp} ...")
    model = onnx.load(args.inp, load_external_data=True)
    replaced = convert(model, fp16=args.fp16)
    kind = "fp16-compute" if args.fp16 else "fp32"
    print(f"replaced {replaced} MatMulNBits node(s) with {kind} MatMul")

    data_file = os.path.basename(args.out) + ".data"
    onnx.save_model(
        model,
        args.out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"wrote {args.out} (+ {data_file})")

    if args.verify:
        verify(args.inp, args.out)


def verify(original_path: str, converted_path: str) -> None:
    import onnxruntime as ort

    print("verifying (CPU EP, random inputs) ...")
    so = ort.SessionOptions()
    a = ort.InferenceSession(original_path, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(converted_path, so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    feeds = {}
    for i in a.get_inputs():
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if "float" in i.type:
            feeds[i.name] = rng.standard_normal(dims).astype(np.float32)
        else:
            feeds[i.name] = np.zeros(dims, dtype=np.int64)

    out_a = a.run(None, feeds)
    out_b = b.run(None, feeds)
    for name, ra, rb in zip([o.name for o in a.get_outputs()], out_a, out_b):
        if ra.dtype.kind == "f":
            diff = float(np.max(np.abs(ra.astype(np.float64) - rb.astype(np.float64))))
            print(f"  {name}: max|Δ| = {diff:.4g}")
        else:
            print(f"  {name}: equal = {np.array_equal(ra, rb)}")


if __name__ == "__main__":
    main()
