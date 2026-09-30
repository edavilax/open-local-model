pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq: usize,
    pub rope: olm::manifest::RopeParameters,
}

impl Config {
    pub fn from_manifest(m: &olm::manifest::Manifest) -> Config {
        Config {
            hidden_size: m.hyperparameters.hidden_size,
            intermediate_size: m.hyperparameters.intermediate_size,
            n_layers: m.hyperparameters.num_hidden_layers,
            n_heads: m.hyperparameters.num_attention_heads,
            n_kv_heads: m.hyperparameters.num_key_value_heads,
            head_dim: m.hyperparameters.head_dim,
            vocab_size: m.hyperparameters.vocab_size,
            rms_norm_eps: m.hyperparameters.rms_norm_eps,
            max_seq: m.metadata.context_window,
            rope: m.hyperparameters.rope.clone(),
        }
    }
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    pub fn group_size(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}
