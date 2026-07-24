use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "experai")]
#[command(about = "Small language model training toolkit", long_about = None)]
pub enum Commands {
    Train {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        data: String,
        #[arg(short, long, default_value = "1e-4")]
        learning_rate: f64,
        #[arg(short, long, default_value = "3")]
        epochs: usize,
    },
    Preprocess {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
    },
    Generate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long, default_value = "50")]
        max_tokens: usize,
    },
}

#[derive(Parser, Debug)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    tracing_subscriber::fmt::init();
    
    match args.command {
        Commands::Train { model, data, learning_rate, epochs } => {
            training::train(&model, &data, learning_rate, epochs)?;
        }
        Commands::Preprocess { input, output } => {
            preprocessing::preprocess(&input, &output)?;
        }
        Commands::Generate { model, prompt, max_tokens } => {
            model::generate(&model, &prompt, max_tokens)?;
        }
    }
    
    Ok(())
}