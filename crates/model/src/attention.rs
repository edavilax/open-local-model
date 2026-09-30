use anyhow::{Ok, Result};

use crate::weights;

use ops::{Backend, RopeTable};
use tensor::Tensor;

pub fn attention(
    x: &Tensor,
    config: Config,
    w: &weights::Attention,
    ro: ReadOnly,
    rw: ReadWrite,
    backend: &impl Backend,
) -> Result<()> {
    // Q
    backend.matmul(x, &w.q, rw.q)?;
    if w.q_bias.is_some() {
        backend.add(rw.q, w.q_bias.as_ref().unwrap())?;
    }
    rw.q.reshape(vec![rw.q.dim(0)?, config.head_dim, rw.q.dim(1)?])?;
    backend.rope(rw.q, ro.rope, config.m_start)?;
    // K
    backend.matmul(x, &w.k, rw.k)?;
    if w.k_bias.is_some() {
        backend.add(rw.k, w.k_bias.as_ref().unwrap())?;
    }
    rw.k.reshape(vec![rw.k.dim(0)?, config.head_dim, rw.k.dim(1)?])?;
    backend.rope(rw.k, ro.rope, config.m_start)?;
    // V
    backend.matmul(x, &w.k, rw.v)?;
    Ok(())
}

pub struct Config {
    m_start: usize,
    head_dim: usize,
}

pub struct ReadOnly<'a> {
    rope: &'a RopeTable,
}

pub struct ReadWrite<'a> {
    q: &'a mut Tensor,
    k: &'a mut Tensor,
    v: &'a mut Tensor,
}
