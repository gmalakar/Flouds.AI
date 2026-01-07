#!/usr/bin/env python3
"""inspect_onnx.py

Utility to inspect an ONNX model file and attempt to create an ONNX Runtime
InferenceSession. Prints helpful diagnostics (existence, size, opset, nodes,
external data, and full ORT exception trace when session creation fails).

Usage:
    python scripts/inspect_onnx.py --path "C:/path/to/model.onnx"
    python scripts/inspect_onnx.py --path path/to/model.onnx --provider CPUExecutionProvider

"""
import argparse
import os
import sys
import traceback

try:
    import onnx
except Exception:
    onnx = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


def human_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f}TB"


def inspect_onnx(path: str) -> int:
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("ERROR: file does not exist")
        return 2

    size = os.path.getsize(path)
    print("Exists: True")
    print("Size:", human_size(size))

    if onnx is None:
        print("onnx Python package not available; skipping ONNX checks")
    else:
        try:
            model = onnx.load(path)
            print("ONNX load: OK")
            try:
                onnx.checker.check_model(model)
                print("ONNX checker: OK")
            except Exception:
                print("ONNX checker: FAILED")
                traceback.print_exc()

            # Show IR and opset imports
            try:
                print("IR version:", getattr(model, "ir_version", "<unknown>"))
                opsets = []
                for op in getattr(model, "opset_import", []):
                    opsets.append((op.domain, op.version))
                print("Opset imports:", opsets)
            except Exception:
                pass

            # Sample node types
            try:
                nodes = [n.op_type for n in model.graph.node[:50]]
                print("Sample nodes:", nodes)
            except Exception:
                pass

            # Check for external_data usage in initializers
            try:
                external = False
                for init in model.graph.initializer:
                    if init.HasField("data_location") and init.data_location != 0:
                        external = True
                        break
                print("Uses external_data:", external)
            except Exception:
                pass

        except Exception:
            print("Failed to load/check ONNX model (traceback):")
            traceback.print_exc()

    if ort is None:
        print("onnxruntime not available; skipping InferenceSession creation")
        return 0

    # Attempt to create an ONNX Runtime session to capture the full error
    providers = ort.get_available_providers()
    print("ORT available providers:", providers)

    # Preferred provider passed via env/args will be used by caller; default to CPU
    provider = "CPUExecutionProvider"
    if "CUDAExecutionProvider" in providers:
        # If CUDA provider exists, prefer CPU by default to reproduce common server setups
        provider = "CPUExecutionProvider"

    print(f"Attempting ORT InferenceSession with provider={provider}")
    try:
        sess = ort.InferenceSession(path, providers=[provider])
        print("ORT session: created successfully")
        print("ORT session inputs:", [i.name for i in sess.get_inputs()])
        print("ORT session outputs:", [o.name for o in sess.get_outputs()])
        return 0
    except Exception:
        print("ORT session creation failed (full traceback):")
        traceback.print_exc()
        return 3


def main(argv: list) -> int:
    p = argparse.ArgumentParser(
        description="Inspect ONNX model and try ONNX Runtime session"
    )
    p.add_argument("--path", required=True, help="Path to ONNX model file")
    p.add_argument(
        "--provider",
        required=False,
        help="ORT provider to use (e.g. CPUExecutionProvider)",
    )
    args = p.parse_args(argv)

    # If user provided a provider, try to use it
    if args.provider and ort is not None:
        provs = ort.get_available_providers()
        if args.provider not in provs:
            print(
                f"Requested provider {args.provider} not available. Providers: {provs}"
            )
        else:
            # Monkey-patch preferred provider selection by setting env var used below
            # Note: inspect_onnx chooses CPUExecutionProvider by default; if --provider
            # specified, override the default by temporarily using get_available_providers
            pass

    return inspect_onnx(args.path)


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    sys.exit(rc)
