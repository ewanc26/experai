#!/usr/bin/env node
// MCP server exposing the experai training toolkit.
//
// Each tool shells out to the `experai` binary (built for macOS/Metal) and
// returns its output. Set EXPERAI_BIN to override the binary path.
//
// Install dependencies:  cd mcp && npm install
// Run:                   node mcp/experai-server.mjs

import { execFile } from "node:child_process"
import { existsSync } from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"
import { z } from "zod"

const execFileAsync = promisify(execFile)

const VERSION = "0.3.0"

const BUILD_HINT =
  "experai binary not found. Build it first with:\n" +
  "  cargo build --release --no-default-features --features metal"

function resolveBinary() {
  const override = process.env.EXPERAI_BIN
  if (override) return override
  for (const candidate of [
    path.resolve(import.meta.dirname, "..", "target", "release", "experai"),
    path.resolve(import.meta.dirname, "..", "target", "debug", "experai"),
  ]) {
    if (existsSync(candidate)) return candidate
  }
  return "experai"
}

async function runExperai(args) {
  const bin = resolveBinary()
  try {
    const { stdout, stderr } = await execFileAsync(bin, args, {
      maxBuffer: 64 * 1024 * 1024,
    })
    const out = stdout.trim()
    if (out) return out
    const err = stderr.trim()
    return err || "(no output)"
  } catch (err) {
    if (err.code === "ENOENT") return BUILD_HINT
    const stderr = (err.stderr ?? "").trim()
    return `experai failed: ${err.message}${stderr ? `\n${stderr}` : ""}`
  }
}

function text(result) {
  return { content: [{ type: "text", text: result }] }
}

const server = new McpServer({
  name: "experai",
  version: VERSION,
})

server.registerTool(
  "support",
  {
    title: "Support Experai",
    description: "Print the sponsor links (Ko-fi, GitHub Sponsors) for the experai project.",
    inputSchema: {},
  },
  async () => text(await runExperai(["--support"])),
)

server.registerTool(
  "commands",
  {
    title: "List experai commands",
    description: "List the subcommands available in the experai CLI.",
    inputSchema: {},
  },
  async () => text(await runExperai(["--json"])),
)

server.registerTool(
  "generate",
  {
    title: "Generate text",
    description:
      "Generate text completions using a trained experai model checkpoint. Returns the generated text.",
    inputSchema: {
      model: z.string().describe("Path to the model checkpoint directory."),
      prompt: z.string().describe("Prompt string to condition generation."),
      max_tokens: z.number().int().positive().optional().describe("Maximum tokens to generate (default 50)."),
      temperature: z.number().positive().optional().describe("Sampling temperature (default 0.7)."),
      top_k: z.number().int().nonnegative().optional().describe("Top-k cutoff (default 50)."),
      top_p: z.number().gt(0).lte(1).optional().describe("Nucleus threshold (default 0.95)."),
      tokenizer: z.string().optional().describe("Path to the tokenizer JSON file."),
    },
  },
  async (args) => {
    const cliArgs = ["generate", "--json", "--model", args.model, "--prompt", args.prompt]
    if (args.max_tokens !== undefined) cliArgs.push("--max-tokens", String(args.max_tokens))
    if (args.temperature !== undefined) cliArgs.push("--temperature", String(args.temperature))
    if (args.top_k !== undefined) cliArgs.push("--top-k", String(args.top_k))
    if (args.top_p !== undefined) cliArgs.push("--top-p", String(args.top_p))
    if (args.tokenizer !== undefined) cliArgs.push("--tokenizer", args.tokenizer)
    const result = await runExperai(cliArgs)
    try {
      const parsed = JSON.parse(result)
      if (parsed.text) return text(parsed.text)
    } catch {
      // not JSON (e.g. error message) — pass through
    }
    return text(result)
  },
)

server.registerTool(
  "train",
  {
    title: "Train a model",
    description:
      "Train an experai model on a tokenized JSONL dataset. Long-running; the client should allow generous timeouts.",
    inputSchema: {
      model: z.string().describe("Pretrained model name or path (e.g. gpt2)."),
      data: z.string().describe("Path to the training dataset directory/file."),
      output_dir: z.string().describe("Directory to write checkpoints and logs."),
      lr: z.number().positive().optional().describe("Learning rate (default 0.0005)."),
      epochs: z.number().int().positive().optional().describe("Training epochs (default 3)."),
      batch_size: z.number().int().positive().optional().describe("Micro-batch size per GPU (default 4)."),
      grad_accum: z.number().int().positive().optional().describe("Gradient accumulation steps (default 8)."),
      resume: z.boolean().optional().describe("Resume from the latest checkpoint in output_dir."),
    },
  },
  async (args) => {
    const cliArgs = ["train", "--model", args.model, "--data", args.data, "--output-dir", args.output_dir]
    if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
    if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
    if (args.batch_size !== undefined) cliArgs.push("--batch-size", String(args.batch_size))
    if (args.grad_accum !== undefined) cliArgs.push("--grad-accum", String(args.grad_accum))
    if (args.resume) cliArgs.push("--resume")
    return text(await runExperai(cliArgs))
  },
)

server.registerTool(
  "preprocess",
  {
    title: "Preprocess data",
    description: "Clean and tokenize raw text data into a training-ready dataset.",
    inputSchema: {
      input: z.string().optional().describe("Input file or directory (stdin when omitted)."),
      output: z.string().describe("Output path for the processed dataset."),
      tokenizer: z.string().optional().describe("Path to the tokenizer JSON file."),
      max_length: z.number().int().positive().optional().describe("Maximum sequence length in tokens (default 512)."),
      clean: z.boolean().optional().describe("Apply text cleaning (lowercase, strip URLs, etc.)."),
      dedupe: z.boolean().optional().describe("Remove duplicate sequences from the output."),
    },
  },
  async (args) => {
    const cliArgs = ["preprocess", "--output", args.output]
    if (args.input !== undefined) cliArgs.push("--input", args.input)
    if (args.tokenizer !== undefined) cliArgs.push("--tokenizer", args.tokenizer)
    if (args.max_length !== undefined) cliArgs.push("--max-length", String(args.max_length))
    if (args.clean) cliArgs.push("--clean")
    if (args.dedupe) cliArgs.push("--dedupe")
    return text(await runExperai(cliArgs))
  },
)

server.registerTool(
  "package",
  {
    title: "Package a checkpoint",
    description: "Export a training checkpoint to GGUF format for use with LM Studio.",
    inputSchema: {
      checkpoint: z.string().describe("Path to the checkpoint directory."),
      name: z.string().describe("Name for the exported GGUF file (without extension)."),
      dtype: z.enum(["f16", "q8_0", "q4_0"]).optional().describe("Quantization dtype (default f16)."),
      lmstudio_dir: z.string().optional().describe("Optional LM Studio models directory to copy into."),
    },
  },
  async (args) => {
    const cliArgs = ["package", "--checkpoint", args.checkpoint, "--name", args.name]
    if (args.dtype !== undefined) cliArgs.push("--dtype", args.dtype)
    if (args.lmstudio_dir !== undefined) cliArgs.push("--lmstudio-dir", args.lmstudio_dir)
    return text(await runExperai(cliArgs))
  },
)

server.registerTool(
  "at_protocol",
  {
    title: "Train from AT Protocol",
    description:
      "Fetch posts from an AT Protocol PDS (e.g. Bluesky) and train a model on them. Long-running.",
    inputSchema: {
      handle: z.string().describe("AT Protocol handle to fetch posts from."),
      max_samples: z.number().int().positive().optional().describe("Maximum number of posts to fetch (default 1000)."),
      output_dir: z.string().describe("Directory to save the fetched dataset and checkpoints."),
      model_name: z.string().optional().describe("Base model name or path (default gpt2)."),
      lr: z.number().positive().optional().describe("Learning rate (default 0.0005)."),
      epochs: z.number().int().positive().optional().describe("Training epochs (default 3)."),
      batch_size: z.number().int().positive().optional().describe("Micro-batch size per GPU (default 4)."),
    },
  },
  async (args) => {
    const cliArgs = ["at-protocol", "--handle", args.handle, "--output-dir", args.output_dir]
    if (args.max_samples !== undefined) cliArgs.push("--max-samples", String(args.max_samples))
    if (args.model_name !== undefined) cliArgs.push("--model-name", args.model_name)
    if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
    if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
    if (args.batch_size !== undefined) cliArgs.push("--batch-size", String(args.batch_size))
    return text(await runExperai(cliArgs))
  },
)

server.registerTool(
  "jetstream_train",
  {
    title: "Train from Jetstream",
    description:
      "Stream posts from the AT Protocol Jetstream relay and train on them. Long-running.",
    inputSchema: {
      max_samples: z.number().int().positive().optional().describe("Stop collecting after this many posts (default 10000)."),
      max_duration_secs: z.number().int().positive().optional().describe("Stop collecting after this many seconds (default 3600)."),
      output_dir: z.string().describe("Directory to save checkpoints."),
      model_name: z.string().optional().describe("Base model name or path (default gpt2)."),
      lr: z.number().positive().optional().describe("Learning rate (default 0.0005)."),
      epochs: z.number().int().positive().optional().describe("Training epochs (default 3)."),
      batch_size: z.number().int().positive().optional().describe("Micro-batch size per GPU (default 4)."),
    },
  },
  async (args) => {
    const cliArgs = ["jetstream-train", "--output-dir", args.output_dir]
    if (args.max_samples !== undefined) cliArgs.push("--max-samples", String(args.max_samples))
    if (args.max_duration_secs !== undefined) cliArgs.push("--max-duration-secs", String(args.max_duration_secs))
    if (args.model_name !== undefined) cliArgs.push("--model-name", args.model_name)
    if (args.lr !== undefined) cliArgs.push("--lr", String(args.lr))
    if (args.epochs !== undefined) cliArgs.push("--epochs", String(args.epochs))
    if (args.batch_size !== undefined) cliArgs.push("--batch-size", String(args.batch_size))
    return text(await runExperai(cliArgs))
  },
)

const transport = new StdioServerTransport()
await server.connect(transport)
