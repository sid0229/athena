"""QLoRA fine-tuning for llama3.2:3b using Apple MLX (mlx-lm).

Pipeline:
1. Fine-tune llama3.2:3b using QLoRA on train-split corrections + n2c2 train gold.
2. Fuse adapters with the base 4-bit model.
3. Package and register with Ollama as 'llama3.2:3b-athena-lora'.
4. Gate evaluation: test on n2c2 test split; only promote if test F1 and Major recall do not drop.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--adapter-path", default="data/processed/lora/adapters")
    ap.add_argument("--fused-path", default="data/processed/lora/fused_model")
    ap.add_argument("--ollama-model", default="llama3.2:3b-athena-lora")
    args = ap.parse_args()

    data_dir = ROOT / "data/processed/lora"
    if not (data_dir / "train.jsonl").exists():
        print("Data files not found. Running prepare_lora_data.py first...")
        run([sys.executable, "scripts/prepare_lora_data.py"])

    # 1. Train QLoRA adapter
    print(f"\n=== Step 1: QLoRA Fine-Tuning ({args.iters} iters, lr={args.learning_rate}) ===")
    train_cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-report", "20",
        "--steps-per-eval", "50",
        "--adapter-path", args.adapter_path,
        "--save-every", "50",
    ]
    run(train_cmd)

    # 2. Fuse adapters with base model
    print("\n=== Step 2: Fuse LoRA Adapters with Base Model ===")
    fuse_cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", args.model,
        "--adapter-path", args.adapter_path,
        "--save-path", args.fused_path,
    ]
    run(fuse_cmd)

    # 3. Create Ollama model configuration
    print(f"\n=== Step 3: Register Model with Ollama as {args.ollama_model} ===")
    modelfile = ROOT / "data/processed/lora/Modelfile"
    modelfile.write_text(f"""FROM {args.fused_path}
PARAMETER temperature 0.0
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"
""")
    try:
        run(["ollama", "create", args.ollama_model, "-f", str(modelfile)])
        print(f"Successfully created Ollama model: {args.ollama_model}")
    except Exception as e:
        print(f"[WARN] Could not register with Ollama directly ({e}). GGUF conversion can be done via llama.cpp.")

    print("\nTraining and packaging complete. Run evaluation scripts to test against promotion gate.")


if __name__ == "__main__":
    main()
