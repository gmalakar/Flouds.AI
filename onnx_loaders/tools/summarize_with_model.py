"""
Summarize text using either an exported ONNX model folder (Optimum ORTModel)
or a Hugging Face Transformers model name/path.

Usage examples (PowerShell):

# Use exported ONNX folder
python .\onnx_loaders\tools\summarize_with_model.py --model_dir "onnx_loaders/onnx/models/s2s/bart-large-cnn" --text "Long article text here..."

# Use HF model id
python .\onnx_loaders\tools\summarize_with_model.py --model_dir "facebook/bart-large-cnn" --text_file .\examples\article.txt --max_length 100

The script will try to load an Optimum ORTModel from `--model_dir` first (if Optimum is installed
and the path looks like an exported ONNX folder). If that fails it falls back to the
standard Hugging Face `AutoModelForSeq2SeqLM` implementation.

"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize text with HF or exported ONNX model"
    )
    parser.add_argument(
        "--model_dir", required=True, help="Local model directory or HF model id"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text", help="Text to summarize (wrap in quotes)")
    grp.add_argument(
        "--text_file", help="Path to a text file containing the input to summarize"
    )
    parser.add_argument(
        "--max_length", type=int, default=80, help="Maximum length of generated summary"
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Beam size for generation"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run on (cpu or cuda)"
    )
    parser.add_argument(
        "--use_onnx_only",
        action="store_true",
        help="Require ONNX backend; fail if Optimum/ORTModel not available",
    )
    parser.add_argument(
        "--decoder_variant",
        choices=["single", "merged", "both"],
        default="single",
        help="Which decoder variant to use when model_dir contains both 'decoder_model.onnx' and 'decoder_model_merged.onnx'",
    )

    args = parser.parse_args()

    model_dir = args.model_dir
    if args.text:
        text = args.text
    else:
        tf = Path(args.text_file)
        if not tf.exists():
            print(f"Text file not found: {tf}")
            sys.exit(2)
        text = tf.read_text(encoding="utf-8")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Failed to load tokenizer from '{model_dir}': {e}")
        sys.exit(3)

    # Try to load an Optimum ONNX ORTModel first (preferred when model_dir contains exported ONNX)
    model = None
    backend = None

    # Helper: attempt to load ORTModel from a path
    def try_load_ort_model(path):
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            return ORTModelForSeq2SeqLM.from_pretrained(path, use_cache=True)
        except Exception:
            return None

    # If user requested 'merged' or 'both', create temporary copies with the
    # decoder file replaced and load ORTModel for each variant.
    tmp_dirs = []
    try:
        import shutil
        from tempfile import mkdtemp

        if args.decoder_variant in ("merged", "both"):
            variants = (
                ["merged"] if args.decoder_variant == "merged" else ["merged", "single"]
            )
            ort_models = {}
            for var in variants:
                tmp = mkdtemp(prefix="onnx_model_")
                tmp_dirs.append(tmp)
                # copy contents of model_dir into tmp
                try:
                    if Path(model_dir).is_dir():
                        for item in Path(model_dir).iterdir():
                            dest = Path(tmp) / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest)
                            else:
                                shutil.copy2(item, dest)
                    else:
                        # model_dir not a directory — skip
                        pass
                except Exception:
                    try:
                        shutil.copytree(model_dir, tmp, dirs_exist_ok=True)
                    except Exception:
                        pass

                src_single = Path(model_dir) / "decoder_model.onnx"
                src_merged = Path(model_dir) / "decoder_model_merged.onnx"
                tgt = Path(tmp) / "decoder_model.onnx"
                try:
                    if var == "merged" and src_merged.exists():
                        shutil.copy2(src_merged, tgt)
                    elif var == "single" and src_single.exists():
                        shutil.copy2(src_single, tgt)
                except Exception:
                    pass

                ort = try_load_ort_model(tmp)
                if ort is not None:
                    ort_models[var] = ort

            if ort_models:
                model = ort_models
                backend = "onnx"
        else:
            # decoder_variant == 'single' -> try loading ORTModel from model_dir
            ort = try_load_ort_model(model_dir)
            if ort is not None:
                model = ort
                backend = "onnx"
    except Exception:
        model = None

    # If we couldn't load an ORT model and ONNX-only was requested, exit.
    if model is None:
        if args.use_onnx_only:
            print(
                "ONNX backend required but could not load Optimum ORTModel from model_dir"
            )
            sys.exit(4)
        # Fallback to HF model
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM

            print("Falling back to Hugging Face Transformers model")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(args.device)
            backend = "hf"
        except Exception as e:
            print(f"Failed to load model from '{model_dir}': {e}")
            sys.exit(5)

    # Prepare inputs
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    if backend == "onnx":
        # ORTModelForSeq2SeqLM.generate accepts torch tensors like HF model
        try:
            if isinstance(model, dict):
                results = {}
                for var, ort_model in model.items():
                    try:
                        outputs = ort_model.generate(
                            **inputs,
                            max_length=args.max_length,
                            num_beams=args.num_beams,
                        )
                        results[var] = tokenizer.batch_decode(
                            outputs, skip_special_tokens=True
                        )[0]
                    except Exception as e:
                        results[var] = f"generation failed: {e}"
                for var, summ in results.items():
                    print(f"\n--- Summary ({var}) ---\n")
                    print(summ)
            else:
                outputs = model.generate(
                    **inputs, max_length=args.max_length, num_beams=args.num_beams
                )
                summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                print("\n--- Summary ---\n")
                print(summary)
        except Exception as e:
            print(f"ONNX model generation failed: {e}")
            sys.exit(6)
    else:
        # HF model path
        try:
            # Ensure tensors are on correct device
            device = torch.device(
                args.device
                if torch.cuda.is_available() and args.device.startswith("cuda")
                else "cpu"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.to(device)
            outputs = model.generate(
                **inputs, max_length=args.max_length, num_beams=args.num_beams
            )
            summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"HF model generation failed: {e}")
            sys.exit(7)

    # HF branch prints summary into `summary` above; ONNX branch already printed.
    if backend != "onnx":
        print("\n--- Summary ---\n")
        print(summary)

    # Cleanup temporary directories if any
    try:
        if "tmp_dirs" in locals() and tmp_dirs:
            import shutil

            for d in tmp_dirs:
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
