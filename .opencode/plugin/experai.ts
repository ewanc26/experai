// opencode plugin exposing the experai training toolkit as first-class tools.
//
// Each tool shells out to the `experai` binary (built for macOS/Metal) and
// returns the structured output. Set EXPERAI_BIN to override the binary path.

import { execFile } from "node:child_process"
import { existsSync } from "node:fs"
import { promisify } from "node:util"
import { tool, type Plugin } from "@opencode-ai/plugin"

const execFileAsync = promisify(execFile)

const BUILD_HINT =
  "experai binary not found. Build it first with:\n" +
  "  cargo build --release --no-default-features --features metal"

function resolveBinary(): string {
  const override = process.env.EXPERAI_BIN
  if (override) return override
  // Prefer the release build in the repo, then debug, then PATH.
  for (const candidate of [
    "./target/release/experai",
    "./target/debug/experai",
  ]) {
    if (existsSync(candidate)) return candidate
  }
  return "experai"
}

async function runExperai(
  args: string[],
  cwd?: string,
): Promise<string> {
  const bin = resolveBinary()
  try {
    const { stdout, stderr } = await execFileAsync(bin, args, {
      cwd,
      maxBuffer: 64 * 1024 * 1024,
      timeout: 0,
    })
    const out = stdout.trim()
    if (out) return out
    const err = stderr.trim()
    return err || "(no output)"
  } catch (err: unknown) {
    const e = err as NodeJS.ErrnoException
    if (e.code === "ENOENT") return BUILD_HINT
    return `experai failed: ${e.message}\n${(e as { stderr?: string }).stderr ?? ""}`
  }
}

export default (async () => {
  return {
    tool: {
      experai_support: tool({
        description:
          "Print the sponsor links (Ko-fi, GitHub Sponsors) for the experai project.",
        args: {},
        async execute() {
          return await runExperai(["--support"])
        },
      }),

      experai_commands: tool({
        description:
          "List the subcommands and options available in the experai CLI.",
        args: {},
        async execute() {
          return await runExperai(["--json"])
        },
      }),

      experai_generate: tool({
        description:
          "Generate text completions using a trained experai model checkpoint.",
        args: {
          model: tool.schema
            .string()
            .describe("Path to the model checkpoint directory."),
          prompt: tool.schema
            .string()
            .describe("Prompt string to condition generation."),
          max_tokens: tool.schema
            .number()
            .optional()
            .describe("Maximum number of tokens to generate (default 50)."),
          temperature: tool.schema
            .number()
            .optional()
            .describe("Sampling temperature (default 0.7)."),
          top_k: tool.schema
            .number()
            .optional()
            .describe("Top-k sampling cutoff (default 50)."),
          top_p: tool.schema
            .number()
            .optional()
            .describe("Nucleus sampling threshold (default 0.95)."),
          tokenizer: tool.schema
            .string()
            .optional()
            .describe("Path to the tokenizer JSON file."),
        },
        async execute(args) {
          const cliArgs = [
            "generate",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
          ]
          if (args.max_tokens !== undefined)
            cliArgs.push("--max-tokens", String(args.max_tokens))
          if (args.temperature !== undefined)
            cliArgs.push("--temperature", String(args.temperature))
          if (args.top_k !== undefined) cliArgs.push("--top-k", String(args.top_k))
          if (args.top_p !== undefined) cliArgs.push("--top-p", String(args.top_p))
          if (args.tokenizer !== undefined)
            cliArgs.push("--tokenizer", args.tokenizer)
          return await runExperai(cliArgs)
        },
      }),

      experai_train: tool({
        description:
          "Train an experai model on a tokenized JSONL dataset. Long-running.",
        args: {
          model: tool.schema
            .string()
            .describe("Pretrained model name or path (e.g. gpt2)."),
          data: tool.schema
            .string()
            .describe("Path to the training dataset directory/file."),
          output_dir: tool.schema
            .string()
            .describe("Directory to write checkpoints and logs."),
          lr: tool.schema.number().optional().describe("Learning rate (default 0.0005)."),
          epochs: tool.schema.number().optional().describe("Training epochs (default 3)."),
          batch_size: tool.schema
            .number()
            .optional()
            .describe("Micro-batch size per GPU (default 4)."),
          grad_accum: tool.schema
            .number()
            .optional()
            .describe("Gradient accumulation steps (default 8)."),
          resume: tool.schema
            .boolean()
            .optional()
            .describe("Resume from the latest checkpoint in output_dir."),
        },
        async execute(args) {
          const cliArgs = [
            "train",
            "--model",
            args.model,
            "--data",
            args.data,
            "--output-dir",
            args.output_dir,
          ]
          if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
          if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
          if (args.batch_size !== undefined)
            cliArgs.push("--batch-size", String(args.batch_size))
          if (args.grad_accum !== undefined)
            cliArgs.push("--grad-accum", String(args.grad_accum))
          if (args.resume) cliArgs.push("--resume")
          return await runExperai(cliArgs)
        },
      }),

      experai_preprocess: tool({
        description:
          "Clean and tokenize raw text data into a training-ready dataset.",
        args: {
          input: tool.schema
            .string()
            .optional()
            .describe("Input file or directory (stdin when omitted)."),
          output: tool.schema
            .string()
            .describe("Output path for the processed dataset."),
          tokenizer: tool.schema
            .string()
            .optional()
            .describe("Path to the tokenizer JSON file."),
          max_length: tool.schema
            .number()
            .optional()
            .describe("Maximum sequence length in tokens (default 512)."),
          clean: tool.schema
            .boolean()
            .optional()
            .describe("Apply text cleaning (lowercase, strip URLs, etc.)."),
          dedupe: tool.schema
            .boolean()
            .optional()
            .describe("Remove duplicate sequences from the output."),
        },
        async execute(args) {
          const cliArgs = ["preprocess", "--output", args.output]
          if (args.input !== undefined) cliArgs.push("--input", args.input)
          if (args.tokenizer !== undefined)
            cliArgs.push("--tokenizer", args.tokenizer)
          if (args.max_length !== undefined)
            cliArgs.push("--max-length", String(args.max_length))
          if (args.clean) cliArgs.push("--clean")
          if (args.dedupe) cliArgs.push("--dedupe")
          return await runExperai(cliArgs)
        },
      }),

      experai_package: tool({
        description:
          "Export a training checkpoint to GGUF format for use with LM Studio.",
        args: {
          checkpoint: tool.schema
            .string()
            .describe("Path to the checkpoint directory."),
          name: tool.schema
            .string()
            .describe("Name for the exported GGUF file (without extension)."),
          dtype: tool.schema
            .string()
            .optional()
            .describe("Quantization dtype: f16, q8_0 or q4_0 (default f16)."),
          lmstudio_dir: tool.schema
            .string()
            .optional()
            .describe("Optional LM Studio models directory to copy into."),
        },
        async execute(args) {
          const cliArgs = [
            "package",
            "--checkpoint",
            args.checkpoint,
            "--name",
            args.name,
          ]
          if (args.dtype !== undefined) cliArgs.push("--dtype", args.dtype)
          if (args.lmstudio_dir !== undefined)
            cliArgs.push("--lmstudio-dir", args.lmstudio_dir)
          return await runExperai(cliArgs)
        },
       }),

      experai_publish_weight: tool({
        description:
          "Publish trained model weights to an AT Protocol repository under click.croft.experai.weight. The password can be provided via EXPERAI_ATP_PASSWORD env var.",
        args: {
          checkpoint: tool.schema
            .string()
            .describe("Path to the checkpoint directory (must contain .safetensors and model_config.json)."),
          handle: tool.schema
            .string()
            .describe("AT Protocol handle or DID of the owning account."),
          password: tool.schema
            .string()
            .optional()
            .describe("Password or app password (use EXPERAI_ATP_PASSWORD env var instead)."),
          pds_url: tool.schema
            .string()
            .optional()
            .describe("Personal Data Server URL (default https://bsky.social)."),
          rkey: tool.schema
            .string()
            .optional()
            .describe("Record key (auto-generated when omitted)."),
          collection: tool.schema
            .string()
            .optional()
            .describe("Collection NSID (default click.croft.experai.weight)."),
          chunk_size: tool.schema
            .number()
            .optional()
            .describe("Chunk size in bytes for blob uploads (default 1000000)."),
          name: tool.schema
            .string()
            .describe("Human-readable name for the weight record."),
          description: tool.schema
            .string()
            .optional()
            .describe("Optional description."),
        },
        async execute(args) {
          const cliArgs = [
            "publish-weight",
            "--checkpoint",
            args.checkpoint,
            "--handle",
            args.handle,
            "--name",
            args.name,
          ]
          if (args.password !== undefined)
            cliArgs.push("--password", args.password)
          if (args.pds_url !== undefined)
            cliArgs.push("--pds-url", args.pds_url)
          if (args.rkey !== undefined) cliArgs.push("--rkey", args.rkey)
          if (args.collection !== undefined)
            cliArgs.push("--collection", args.collection)
          if (args.chunk_size !== undefined)
            cliArgs.push("--chunk-size", String(args.chunk_size))
          if (args.description !== undefined)
            cliArgs.push("--description", args.description)
          return await runExperai(cliArgs)
        },
      }),

      experai_atprotocol: tool({
        description:
          "Fetch posts from an AT Protocol PDS (e.g. Bluesky) and train a model on them.",
        args: {
          handle: tool.schema
            .string()
            .describe("AT Protocol handle to fetch posts from."),
          max_samples: tool.schema
            .number()
            .optional()
            .describe("Maximum number of posts to fetch (default 1000)."),
          output_dir: tool.schema
            .string()
            .describe("Directory to save the fetched dataset and checkpoints."),
          model_name: tool.schema
            .string()
            .optional()
            .describe("Base model name or path (default gpt2)."),
          lr: tool.schema.number().optional().describe("Learning rate (default 0.0005)."),
          epochs: tool.schema.number().optional().describe("Training epochs (default 3)."),
          batch_size: tool.schema
            .number()
            .optional()
            .describe("Micro-batch size per GPU (default 4)."),
        },
        async execute(args) {
          const cliArgs = [
            "at-protocol",
            "--handle",
            args.handle,
            "--output-dir",
            args.output_dir,
          ]
          if (args.max_samples !== undefined)
            cliArgs.push("--max-samples", String(args.max_samples))
          if (args.model_name !== undefined)
            cliArgs.push("--model-name", args.model_name)
          if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
          if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
          if (args.batch_size !== undefined)
            cliArgs.push("--batch-size", String(args.batch_size))
          return await runExperai(cliArgs)
        },
      }),

      experai_jetstream: tool({
        description:
          "Stream posts from the AT Protocol Jetstream relay and train on them.",
        args: {
          max_samples: tool.schema
            .number()
            .optional()
            .describe("Stop collecting after this many posts (default 10000)."),
          max_duration_secs: tool.schema
            .number()
            .optional()
            .describe("Stop collecting after this many seconds (default 3600)."),
          output_dir: tool.schema
            .string()
            .describe("Directory to save checkpoints."),
          model_name: tool.schema
            .string()
            .optional()
            .describe("Base model name or path (default gpt2)."),
          lr: tool.schema.number().optional().describe("Learning rate (default 0.0005)."),
          epochs: tool.schema.number().optional().describe("Training epochs (default 3)."),
          batch_size: tool.schema
            .number()
            .optional()
            .describe("Micro-batch size per GPU (default 4)."),
        },
        async execute(args) {
          const cliArgs = ["jetstream-train", "--output-dir", args.output_dir]
          if (args.max_samples !== undefined)
            cliArgs.push("--max-samples", String(args.max_samples))
          if (args.max_duration_secs !== undefined)
            cliArgs.push("--max-duration-secs", String(args.max_duration_secs))
          if (args.model_name !== undefined)
            cliArgs.push("--model-name", args.model_name)
          if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
          if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
          if (args.batch_size !== undefined)
            cliArgs.push("--batch-size", String(args.batch_size))
          return await runExperai(cliArgs)
        },
      }),
    },
  }
}) satisfies Plugin
