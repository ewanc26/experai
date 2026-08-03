//! CLI entry point for the Experai training toolkit.
//!
//! Parses subcommands via `clap` and dispatches to the corresponding
//! command handlers in [`experai::commands`].

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use experai::logging::init_logger;

/// Top-level CLI parser.
#[derive(Parser, Debug)]
#[command(name = "experai")]
#[command(version = "0.3.0")]
#[command(about = "Small language model training toolkit")]
struct Cli {
    /// Print sponsor links and exit.
    #[arg(long)]
    support: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a model on a tokenized dataset.
    Train {
        /// Path to the pretrained model weights or HuggingFace model ID.
        #[arg(short, long)]
        model: String,
        /// Path to the training dataset directory.
        #[arg(short, long)]
        data: String,
        /// Learning rate for the optimizer.
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        /// Number of training epochs.
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        /// Micro-batch size per GPU.
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        /// Number of gradient accumulation steps before each optimizer update.
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        /// Directory to write checkpoints and logs.
        #[arg(short = 'o', long)]
        output_dir: String,
        /// Path to the tokenizer JSON file.
        #[arg(short = 't', long, default_value = "models/tokenizer.json")]
        tokenizer: String,
        /// Automatically tune batch size based on available VRAM.
        #[arg(long)]
        auto_tune: bool,
        /// Resume training from the latest checkpoint in `output_dir`.
        #[arg(long)]
        resume: bool,
    },
    /// Preprocess raw text data into tokenized training format.
    Preprocess {
        /// Input file or directory (reads stdin when omitted).
        #[arg(short = 'i', long)]
        input: Option<String>,
        /// Output path for the processed dataset.
        #[arg(short = 'o', long)]
        output: String,
        /// Path to the tokenizer JSON file.
        #[arg(short = 't', long)]
        tokenizer: String,
        /// Maximum sequence length in tokens.
        #[arg(short = 'l', long, default_value = "512")]
        max_length: usize,
        /// AT Protocol handle to fetch posts from (skips fetching when omitted).
        #[arg(long)]
        atproto_handle: Option<String>,
        /// AT Protocol collection namespace to fetch.
        #[arg(long, default_value = "app.bsky.feed.post")]
        collection: String,
        /// Remove duplicate sequences from the output.
        #[arg(long)]
        dedupe: bool,
        /// Apply text cleaning (lowercase, strip URLs, etc.).
        #[arg(long)]
        clean: bool,
        /// Specific DIDs to fetch posts from.
        #[arg(short = 'd', long)]
        dids: Vec<String>,
    },
    /// Generate text completions using a trained model.
    Generate {
        /// Path to the model checkpoint directory.
        #[arg(short = 'm', long)]
        model: String,
        /// Prompt string to condition generation.
        #[arg(short = 'p', long)]
        prompt: String,
        /// Maximum number of tokens to generate.
        #[arg(short = 'n', long, default_value = "50")]
        max_tokens: usize,
        /// Sampling temperature (higher = more random).
        #[arg(short = 't', long, default_value = "0.7")]
        temperature: f64,
        /// Top-k: keep only the k most probable tokens at each step.
        #[arg(short = 'k', long, default_value = "50")]
        top_k: usize,
        /// Nucleus sampling: keep tokens whose cumulative probability reaches this threshold.
        #[arg(short = 'P', long, default_value = "0.95")]
        top_p: f64,
        /// Path to the tokenizer JSON file.
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
    },
    /// Fetch posts from an AT Protocol PDS and train a model on them.
    AtProtocol {
        /// Personal Data Server URL.
        #[arg(short = 'u', long, default_value = "https://bsky.social")]
        pds_url: String,
        /// AT Protocol handle to fetch posts from.
        #[arg(short = 'h', long, default_value = "public")]
        handle: String,
        /// Maximum number of posts to fetch.
        #[arg(short = 'n', long, default_value = "1000")]
        max_samples: usize,
        /// Directory to save the fetched dataset.
        #[arg(short = 'o', long)]
        output_dir: String,
        /// Base model name or path to use for training.
        #[arg(short = 'm', long, default_value = "gpt2")]
        model_name: String,
        /// Learning rate for the optimizer.
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        /// Number of training epochs.
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        /// Micro-batch size per GPU.
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        /// Gradient accumulation steps.
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        /// Path to the tokenizer JSON file.
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
    },
    /// Stream posts from the Jetstream relay and train on them.
    JetstreamTrain {
        /// Jetstream WebSocket host.
        #[arg(long, default_value = "jetstream2.us-east.bsky.network")]
        jetstream_host: String,
        /// AT Protocol collections to subscribe to.
        #[arg(short = 'c', long, default_value = "app.bsky.feed.post")]
        collections: Vec<String>,
        /// Specific DIDs to filter on.
        #[arg(short = 'd', long)]
        dids: Vec<String>,
        /// Stop collecting after this many posts.
        #[arg(short = 'n', long, default_value = "10000")]
        max_samples: usize,
        /// Stop collecting after this many seconds.
        #[arg(long, default_value = "3600")]
        max_duration_secs: u64,
        /// Directory to save checkpoints.
        #[arg(short = 'o', long)]
        output_dir: String,
        /// Base model name or path for training.
        #[arg(short = 'm', long, default_value = "gpt2")]
        model_name: String,
        /// Learning rate.
        #[arg(short = 'l', long, default_value = "0.0005")]
        lr: f64,
        /// Number of training epochs.
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        /// Micro-batch size per GPU.
        #[arg(short = 'b', long, default_value = "4")]
        batch_size: usize,
        /// Gradient accumulation steps.
        #[arg(short = 'g', long, default_value = "8")]
        grad_accum: usize,
        /// Path to the tokenizer JSON file.
        #[arg(long, default_value = "models/tokenizer.json")]
        tokenizer: String,
        /// Automatically tune batch size based on available VRAM.
        #[arg(long)]
        auto_tune: bool,
    },
    /// Export a training checkpoint to GGUF format for use with LM Studio.
    Package {
        /// Path to the checkpoint directory.
        #[arg(short = 'c', long)]
        checkpoint: String,
        /// Name for the exported GGUF file (without extension).
        #[arg(short = 'n', long)]
        name: String,
        /// Optional LM Studio models directory to copy the export into.
        #[arg(long)]
        lmstudio_dir: Option<String>,
        /// Quantization dtype for the GGUF export (`f16`, `q8_0`, `q4_0`).
        #[arg(long, default_value = "f16")]
        dtype: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.support {
        println!("Support Experai development:");
        println!("  Ko-fi: https://ko-fi.com/ewancroft");
        println!("  GitHub Sponsors: https://github.com/sponsors/ewanc26");
        return Ok(());
    }

    // No subcommand and no --support: show help instead of doing nothing.
    let Some(command) = cli.command else {
        Cli::command().print_help()?;
        println!();
        return Ok(());
    };

    // Allow overriding the log level via environment variable.
    let log_level = std::env::var("EXPERAI_LOG_LEVEL").ok();
    init_logger(log_level, false)?;

    // Dispatch to the appropriate command handler.
    match command {
        Commands::Train {
            model,
            data,
            lr,
            epochs,
            batch_size,
            grad_accum,
            output_dir,
            tokenizer,
            auto_tune,
            resume,
        } => {
            experai::commands::train::run(
                model, data, lr, epochs, batch_size, grad_accum, output_dir, tokenizer, auto_tune,
                resume,
            )?;
        }
        Commands::Preprocess {
            input,
            output,
            tokenizer,
            max_length,
            atproto_handle,
            collection,
            dedupe,
            clean,
            dids,
        } => {
            experai::commands::preprocess::run(
                input,
                output,
                tokenizer,
                max_length,
                atproto_handle,
                collection,
                dedupe,
                clean,
                dids,
            )?;
        }
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            tokenizer,
        } => {
            experai::commands::generate::run(
                model,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                tokenizer,
            )?;
        }
        Commands::AtProtocol {
            pds_url,
            handle,
            max_samples,
            output_dir,
            model_name,
            lr,
            epochs,
            batch_size,
            grad_accum,
            tokenizer,
        } => {
            experai::commands::at_protocol::run(
                pds_url,
                handle,
                max_samples,
                output_dir,
                model_name,
                lr,
                epochs,
                batch_size,
                grad_accum,
                tokenizer,
            )?;
        }
        Commands::JetstreamTrain {
            jetstream_host,
            collections,
            dids,
            max_samples,
            max_duration_secs,
            output_dir,
            model_name,
            lr,
            epochs,
            batch_size,
            grad_accum,
            tokenizer,
            auto_tune,
        } => {
            experai::commands::jetstream_train::run(
                jetstream_host,
                collections,
                dids,
                max_samples,
                max_duration_secs,
                output_dir,
                model_name,
                lr,
                epochs,
                batch_size,
                grad_accum,
                tokenizer,
                auto_tune,
            )?;
        }
        Commands::Package {
            checkpoint,
            name,
            lmstudio_dir,
            dtype,
        } => {
            experai::commands::package::run(checkpoint, name, lmstudio_dir, dtype)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    #[test]
    fn support_flag_parses_without_subcommand() {
        let cli = Cli::try_parse_from(["experai", "--support"]).unwrap();
        assert!(cli.support);
        assert!(cli.command.is_none());
    }

    #[test]
    fn support_flag_with_subcommand() {
        let cli = Cli::try_parse_from([
            "experai",
            "--support",
            "train",
            "-m",
            "gpt2",
            "-d",
            "data",
            "-o",
            "out",
        ])
        .unwrap();
        assert!(cli.support);
        assert!(cli.command.is_some());
    }

    #[test]
    fn no_args_allows_no_subcommand() {
        let cli = Cli::try_parse_from(["experai"]).unwrap();
        assert!(!cli.support);
        assert!(cli.command.is_none());
    }

    #[test]
    fn subcommands_are_optional_without_support() {
        let cli = Cli::try_parse_from(["experai", "generate", "-m", "out", "-p", "hi"]).unwrap();
        assert!(!cli.support);
        assert!(cli.command.is_some());
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        let err = Cli::try_parse_from(["experai", "bogus"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn support_flag_requires_no_other_arguments() {
        let cli = Cli::try_parse_from(["experai", "--support"]).unwrap();
        assert!(cli.support);
    }
}
