use experai::preprocessing::PreprocessConfig;
use experai::training::{compute_perplexity, TrainingConfig};
use experai::utils::{check_value_in_range, GradientAccumulator};

#[test]
fn gradient_accumulator_steps_correctly() {
    let mut accum = GradientAccumulator::new(3);
    assert!(!accum.step());
    assert!(!accum.step());
    assert!(accum.step());
    assert!(!accum.step());
}

#[test]
fn gradient_accumulator_reset_works() {
    let mut accum = GradientAccumulator::new(2);
    assert!(!accum.step());
    assert!(accum.step());
    accum.reset();
    assert!(!accum.step());
}

#[test]
fn compute_perplexity_is_exp() {
    let perp = compute_perplexity(0.0);
    assert!((perp - 1.0).abs() < 0.001);
}

#[test]
fn check_value_in_range_works() {
    assert!(check_value_in_range(5, 0, 10));
    assert!(!check_value_in_range(-1, 0, 10));
    assert!(check_value_in_range(0, 0, 10));
    assert!(check_value_in_range(10, 0, 10));
}

#[test]
fn training_config_has_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.gradient_accumulation_steps, 8);
    assert!(config.enable_load_monitoring);
    assert_eq!(config.max_cpu_load, 0.80);
    // Swap avoidance: never allow swap, keep 2 GB free.
    assert_eq!(config.max_swap_usage, 0.0);
    assert_eq!(config.min_free_mem_mb, 2048);
}

#[test]
fn preprocess_config_defaults() {
    let _ = PreprocessConfig::default(); // just verify it constructs
}
