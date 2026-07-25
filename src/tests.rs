#[cfg(test)]
mod tests {
    use crate::data::{DatasetSample, DataCollator};
    use crate::model::{ModelConfig, build_model, causal_mask};
    use crate::training::TrainingConfig;
    use candle_core::{Device, Tensor, DType};
    use candle_nn::{Module, VarBuilder, VarMap};
    use tokenizers::Tokenizer;

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 3);
        assert_eq!(config.learning_rate, 5e-5);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.gradient_accumulation_steps, 8);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
    }

    #[test]
    fn test_data_collator() {
        let collator = DataCollator::new(0, 64);
        let samples = vec![
            DatasetSample {
                text: "hello world".to_string(),
                tokens: vec![5, 10, 2],
                label: None,
                did: None,
            },
            DatasetSample {
                text: "foo bar".to_string(),
                tokens: vec![8, 12],
                label: None,
                did: None,
            },
        ];
        let (input_ids, _attention_mask) = collator.collate(&samples).unwrap();
        let (batch, seq) = input_ids.dims2().unwrap();
        assert_eq!(batch, 2);
        assert_eq!(seq, 3);
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = crate::utils::GradientAccumulator::new(4);
        assert!(!acc.step());
        assert!(!acc.step());
        assert!(!acc.step());
        assert!(acc.step());
        assert!(!acc.step());
    }

    #[test]
    fn test_check_value_in_range() {
        assert!(crate::utils::check_value_in_range(5, 0, 10));
        assert!(!crate::utils::check_value_in_range(15, 0, 10));
        assert!(crate::utils::check_value_in_range(0, 0, 10));
        assert!(crate::utils::check_value_in_range(10, 0, 10));
    }

    #[test]
    fn test_model_forward_pass() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 128,
            max_seq_len: 64,
            hidden_act: "gelu".to_string(),
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        };

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = build_model(&config, vb).expect("failed to build model");

        let input_ids = Tensor::rand(0f32, 100f32, (1, 10), &device)
            .expect("failed to create input")
            .to_dtype(DType::U32)
            .expect("failed to cast to u32");
        let output = model.forward(&input_ids).expect("forward pass failed");

        assert_eq!(output.dims(), &[1, 10, 100]);
        let has_nan = output.to_vec3::<f32>().unwrap().iter().flatten().flatten().any(|v| v.is_nan());
        assert!(!has_nan, "output contains NaN values");
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let seq_len = 5;
        let mask = causal_mask(seq_len, &device).expect("causal_mask failed");

        assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);

        let mask_2d = mask.reshape((seq_len, seq_len)).expect("reshape failed");
        let mask_vec = mask_2d.to_vec2::<f32>().unwrap();
        for i in 0..seq_len {
            for j in 0..seq_len {
                let val = mask_vec[i][j];
                if j > i {
                    assert!(val.is_infinite() && val < 0.0,
                        "expected -inf at ({},{}) but got {}", i, j, val);
                } else {
                    assert_eq!(val, 0.0, "expected 0.0 at ({},{}) but got {}", i, j, val);
                }
            }
        }
    }

    // Preprocessing unit tests

    fn make_preprocessor(config: crate::preprocessing::PreprocessConfig) -> crate::preprocessing::Preprocessor {
        use tokenizers::models::bpe::BPE;
        let tokenizer = Tokenizer::new(BPE::default());
        crate::preprocessing::Preprocessor::new(tokenizer, Some(config))
    }

    fn preprocessor_for(cleanup_fn: impl FnOnce(&mut crate::preprocessing::PreprocessConfig)) -> crate::preprocessing::Preprocessor {
        let mut config = crate::preprocessing::PreprocessConfig {
            lowercase: false,
            remove_extra_whitespace: false,
            remove_html_tags: false,
            remove_urls: false,
            remove_emails: false,
            min_length: 0,
            max_length: 1024,
            dedupe: false,
            clean: false,
        };
        cleanup_fn(&mut config);
        make_preprocessor(config)
    }

    #[test]
    fn test_clean_text_lowercase() {
        let p = preprocessor_for(|c| c.lowercase = true);
        assert_eq!(p.clean_text("Hello WORLD"), "hello world");
        assert_eq!(p.clean_text("FOO"), "foo");
        assert_eq!(p.clean_text("MiXeD"), "mixed");
    }

    #[test]
    fn test_clean_text_remove_html() {
        let p = preprocessor_for(|c| c.remove_html_tags = true);
        assert_eq!(p.clean_text("Hello <b>world</b>!"), "Hello world!");
        assert_eq!(
            p.clean_text("A <div class=\"x\">nested</div> test."),
            "A nested test."
        );
        assert_eq!(
            p.clean_text("<p>Paragraph</p><br/>Line break"),
            "ParagraphLine break"
        );
    }

    #[test]
    fn test_clean_text_remove_urls() {
        let p = preprocessor_for(|c| c.remove_urls = true);
        assert_eq!(
            p.clean_text("Visit https://example.com for more"),
            "Visit [URL] for more"
        );
        assert_eq!(
            p.clean_text("http://a.com and https://b.org/path?q=1"),
            "[URL] and [URL]"
        );
        assert_eq!(p.clean_text("no urls here"), "no urls here");
    }

    #[test]
    fn test_clean_text_remove_emails() {
        let p = preprocessor_for(|c| c.remove_emails = true);
        assert_eq!(
            p.clean_text("Contact user@example.com for info"),
            "Contact [EMAIL] for info"
        );
        assert_eq!(
            p.clean_text("a@b.com and x@y.org"),
            "[EMAIL] and [EMAIL]"
        );
        assert_eq!(p.clean_text("no email here"), "no email here");
    }

    #[test]
    fn test_clean_text_remove_whitespace() {
        let p = preprocessor_for(|c| c.remove_extra_whitespace = true);
        assert_eq!(p.clean_text("hello   world"), "hello world");
        assert_eq!(p.clean_text("  leading and trailing  "), "leading and trailing");
        assert_eq!(p.clean_text("tabs\tand\nnewlines"), "tabs and newlines");
        assert_eq!(p.clean_text("one"), "one");
    }

    // Auto-tune tests

    #[test]
    fn test_hardware_profile_detect() {
        let profile = crate::utils::HardwareProfile::detect();
        assert!(!profile.device_name.is_empty());
        assert!(profile.cpu_cores > 0);
        assert!(profile.ram_mb > 0);
    }

    #[test]
    fn test_auto_tuner_batch_strategy() {
        let profile_24g = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(24000),
            ram_mb: 32768,
            cpu_cores: 8,
            supports_bf16: true,
            supports_fp16: true,
        };
        let params_24g = crate::utils::AutoTuner::recommend(&profile_24g);
        assert_eq!(params_24g.batch_size, 32);

        let profile_8g = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(8000),
            ram_mb: 16384,
            cpu_cores: 4,
            supports_bf16: true,
            supports_fp16: true,
        };
        let params_8g = crate::utils::AutoTuner::recommend(&profile_8g);
        assert_eq!(params_8g.batch_size, 4);
    }

    #[test]
    fn test_auto_tuner_seq_len_strategy() {
        // Test sequence length selection
        let profile_large = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(16000),
            ram_mb: 32768,
            cpu_cores: 8,
            supports_bf16: true,
            supports_fp16: true,
        };
        let params = crate::utils::AutoTuner::recommend(&profile_large);
        assert_eq!(params.max_seq_len, 2048);

        let profile_small = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(4000),
            ram_mb: 8192,
            cpu_cores: 2,
            supports_bf16: false,
            supports_fp16: false,
        };
        let params = crate::utils::AutoTuner::recommend(&profile_small);
        assert_eq!(params.max_seq_len, 512);
        assert_eq!(params.precision, "f32");
    }

    #[test]
    fn test_auto_tuner_precision_selection() {
        let bf16_profile = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(16000),
            ram_mb: 32768,
            cpu_cores: 8,
            supports_bf16: true,
            supports_fp16: false,
        };
        let params = crate::utils::AutoTuner::recommend(&bf16_profile);
        assert_eq!(params.precision, "bf16");

        let fp16_profile = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(16000),
            ram_mb: 32768,
            cpu_cores: 8,
            supports_bf16: false,
            supports_fp16: true,
        };
        let params = crate::utils::AutoTuner::recommend(&fp16_profile);
        assert_eq!(params.precision, "fp16");

        let f32_profile = crate::utils::HardwareProfile {
            device: crate::utils::ComputeDevice::Cpu,
            device_name: "CPU".to_string(),
            vram_mb: Some(16000),
            ram_mb: 32768,
            cpu_cores: 8,
            supports_bf16: false,
            supports_fp16: false,
        };
        let params = crate::utils::AutoTuner::recommend(&f32_profile);
        assert_eq!(params.precision, "f32");
    }
}
