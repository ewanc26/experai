---
description: Run an experai CLI command (train, generate, preprocess, package, opencode, --support, ...)
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
   - `opencode` → summarize the GGUF/Modelfile paths and the model id to
     select in OpenCode (`<provider>/<name>`).
   - if `$ARGUMENTS` is empty, print the CLI help (`--help`).

Examples:

- `/experai --support`
- `/experai generate --model output --prompt "The future of AI is" --max-tokens 50`
- `/experai train --model gpt2 --data data/train.jsonl --output-dir output --epochs 3`
- `/experai train --model gpt2 --data notes.txt --output-dir output` (any text file, CSV, JSON, directory, or `-` for stdin)
- `/experai preprocess --input raw.txt --output data/train.jsonl --tokenizer models/tokenizer.json`
- `/experai opencode --checkpoint output --name my-model --config-path opencode.json`
- `/experai fetch --model gpt2 --what all   # grab tokenizer.json + config.json from HuggingFace`

For JSON output (useful for parsing), pass `--json` after `experai`.
