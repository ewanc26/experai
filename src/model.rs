use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{linear_no_bias, Dropout, Embedding, Linear, Module, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub hidden_act: String,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_seq_len: 2048,
            hidden_act: "gelu".to_string(),
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub norm: candle_nn::RmsNorm,
    pub lm_head: Linear,
    pub dropout: Dropout,
}

pub struct TransformerLayer {
    self_attn: MultiHeadAttention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

pub fn build_model(config: &ModelConfig, vb: VarBuilder) -> Result<TransformerModel> {
    let embedding =
        candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embedding"))?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let layer = TransformerLayer::new(config, vb.pp(format!("layers.{}", i)))?;
        layers.push(layer);
    }

    let norm = candle_nn::rms_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
    let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
    let dropout = Dropout::new(0.1);

    Ok(TransformerModel {
        config: config.clone(),
        embedding,
        layers,
        norm,
        lm_head,
        dropout,
    })
}

impl TransformerLayer {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let self_attn = MultiHeadAttention::new(config, head_dim, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h)?;
        let h = (h + residual)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = (h + residual)?;

        Ok(h)
    }
}

impl MultiHeadAttention {
    fn new(config: &ModelConfig, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?;

        // Causal mask
        let mask = causal_mask(seq_len, x.device())?;
        let scores = (q
            .contiguous()?
            .matmul(&k_t.contiguous()?)?
            .broadcast_add(&mask)?
            * scale)?;

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v.contiguous()?)?;

        // Reshape back to (batch, seq, hidden)
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, hidden_size))?;

        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }
}

pub(crate) fn causal_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    // Create an upper triangular mask of -inf
    let mask_vals: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::from_slice(&mask_vals, (seq_len, seq_len), device)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    Ok(mask)
}

impl Mlp {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = gate.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        let gate_up = (gate * up)?;
        let out = self.down_proj.forward(&gate_up)?;
        Ok(out)
    }
}

impl Module for TransformerModel {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = self.embedding.forward(x)?;
        h = self.dropout.forward(&h, false)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        h = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        Ok(logits)
    }
}
