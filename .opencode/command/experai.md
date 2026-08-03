---
description: Run an experai CLI command (train, generate, preprocess, package, --support, ...)
agent: build
---

Run the experai training toolkit with the arguments typed after `/experai`.

1. If the binary is missing, build it first (macOS/Metal only — never build the
   default `cuda` feature locally):

   ```bash
   cargo build --release --no-default-features --features metal
   ```

2. Execute the CLI, preferring the release binary when it exists:

   ```bash
   ./target/release/experai $ARGUMENTS
   ```

3. Report the result concisely:
   - `--support` → show the Ko-fi and GitHub Sponsors links.
   - `generate` → show the generated text.
   - `train` / `preprocess` / `package` → summarize the outcome (output paths,
     loss, file written).
   - if `$ARGUMENTS` is empty, print the CLI help (`--help`).

Examples:

- `/experai --support`
- `/experai generate --model output --prompt "The future of AI is" --max-tokens 50`
- `/experai train --model gpt2 --data data/train.jsonl --output-dir output --epochs 3`
- `/experai preprocess --input raw.txt --output data/train.jsonl --tokenizer models/tokenizer.json`

For JSON output (useful for parsing), pass `--json` after `experai`.
