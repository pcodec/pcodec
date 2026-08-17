use std::collections::BTreeMap;

use serde::Serialize;
use tabled::Tabled;

#[derive(Serialize)]
pub struct CompressionSummary {
  pub ratio: f64,
  pub total_size: usize,
  pub header_size: usize,
  pub meta_size: usize,
  pub page_size: usize,
  pub footer_size: usize,
  pub unknown_trailing_bytes: usize,
}

#[derive(Tabled)]
pub struct BinSummary {
  pub weight: u32,
  pub lower: String,
  pub offset_bits: u32,
}

#[derive(Serialize)]
pub struct LatentVarSummary {
  pub name: String,
  pub latent_type: String,
  pub n_bins: usize,
  pub ans_size_log: u32,
  pub approx_avg_bits: f64,
  pub bins: String,
}

#[derive(Serialize)]
pub struct ChunkSummary {
  pub idx: usize,
  pub n: usize,
  pub mode: String,
  pub delta_encoding: String,
  // using BTreeMaps to preserve ordering
  pub latent_var: BTreeMap<String, LatentVarSummary>,
}

#[derive(Serialize)]
pub struct Summary {
  pub filename: String,
  pub format_version: String,
  pub number_type: String,
  pub n: usize,
  pub n_chunks: usize,
  pub uncompressed_size: usize,
  pub compressed: CompressionSummary,
  pub chunk: Vec<ChunkSummary>,
}
