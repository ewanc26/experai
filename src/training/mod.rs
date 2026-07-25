/// Training infrastructure: configuration, loss computation, and the main training loop.
pub mod config;
pub mod loss;
pub mod trainer;

pub use config::*;
pub use loss::*;
pub use trainer::*;
