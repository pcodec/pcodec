use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;
use tabled::settings::object::Columns;
use tabled::settings::{Alignment, Modify, Style};
use tabled::{Table, Tabled};

use pco::data_types::{Latent, Number};
use pco::match_latent_enum;
use pco::metadata::{ChunkMeta, DynBins, DynLatent, LatentVarKey};
use pco::standalone::{FileDecompressor, MaybeChunkDecompressor};

use crate::core_handlers::CoreHandlerImpl;
use crate::dtypes::PcoNumber;
use crate::inspect::InspectOpt;
use crate::utils;

pub trait InspectHandler {
  fn inspect(&self, opt: &InspectOpt, bytes: &[u8]) -> Result<()>;
}

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
  weight: u32,
  lower: String,
  offset_bits: u32,
}

#[derive(Serialize)]
pub struct LatentVarSummary {
  name: String,
  n_bins: usize,
  ans_size_log: u32,
  approx_avg_bits: f64,
  bins: String,
}

#[derive(Serialize)]
pub struct ChunkSummary {
  idx: usize,
  n: usize,
  mode: String,
  delta_encoding: String,
  // using BTreeMaps to preserve ordering
  latent_vars: BTreeMap<String, LatentVarSummary>,
}

#[derive(Serialize)]
pub struct Output {
  pub filename: String,
  pub data_type: String,
  pub format_version: String,
  pub n: usize,
  pub n_chunks: usize,
  pub uncompressed_size: usize,
  pub compressed: CompressionSummary,
  pub chunks: Vec<ChunkSummary>,
}

fn measure_bytes_read(src: &[u8], prev_src_len: &mut usize) -> usize {
  let res = *prev_src_len - src.len();
  *prev_src_len = src.len();
  res
}

fn build_latent_var_summaries<T: Number>(meta: &ChunkMeta) -> BTreeMap<String, LatentVarSummary> {
  let describers = T::get_latent_describers(meta);
  let mut summaries = BTreeMap::new();
  for (key, (latent_var_meta, describer)) in meta
    .per_latent_var
    .as_ref()
    .zip_exact(describers)
    .enumerated()
  {
    let unit = describer.latent_units();

    let mut approx_total_bits = 0.0;
    let bin_summaries = match_latent_enum!(
      &latent_var_meta.bins,
      DynBins<L>(bins) => {
        let mut bin_summaries = Vec::new();
        for bin in bins {
          bin_summaries.push(BinSummary {
            weight: bin.weight,
            lower: format!("{}{}", describer.latent(DynLatent::new(bin.lower)), unit),
            offset_bits: bin.offset_bits,
          });
          let weight = bin.weight as f64;
          approx_total_bits += weight * (bin.offset_bits as f64 + latent_var_meta.ans_size_log as f64 - weight.log2());
        }
        bin_summaries
      }
    );
    let n_bins = bin_summaries.len();
    let bins_table = Table::new(bin_summaries)
      .with(Style::rounded())
      .with(Modify::new(Columns::new(0..3)).with(Alignment::right()))
      .to_string();
    let total_weight = (1 << latent_var_meta.ans_size_log) as f64;

    let summary = LatentVarSummary {
      name: describer.latent_var(),
      n_bins,
      ans_size_log: latent_var_meta.ans_size_log,
      approx_avg_bits: approx_total_bits / total_weight,
      bins: bins_table.to_string(),
    };

    let key_name = match key {
      LatentVarKey::Delta => "delta",
      LatentVarKey::Primary => "primary",
      LatentVarKey::Secondary => "secondary",
    };

    summaries.insert(key_name.to_string(), summary);
  }

  summaries
}

impl<T: PcoNumber> InspectHandler for CoreHandlerImpl<T> {
  fn inspect(&self, opt: &InspectOpt, src: &[u8]) -> Result<()> {
    let mut prev_src_len_val = src.len();
    let prev_src_len = &mut prev_src_len_val;
    let (fd, mut src) = FileDecompressor::new(src)?;
    let header_size = measure_bytes_read(src, prev_src_len);

    let mut meta_size = 0;
    let mut page_size = 0;
    let mut footer_size = 0;
    let mut chunk_ns = Vec::new();
    let mut metas = Vec::new();
    let mut void = Vec::new();
    loop {
      // Rather hacky, but first just measure the metadata size,
      // then reread it to measure the page size
      match fd.chunk_decompressor::<T, _>(src)? {
        MaybeChunkDecompressor::Some(cd) => {
          chunk_ns.push(cd.n());
          metas.push(cd.meta().clone());
          meta_size += measure_bytes_read(cd.into_src(), prev_src_len);
        }
        MaybeChunkDecompressor::EndOfData(rest) => {
          src = rest;
          footer_size += measure_bytes_read(src, prev_src_len);
          break;
        }
      }

      match fd.chunk_decompressor::<T, _>(src)? {
        MaybeChunkDecompressor::Some(mut cd) => {
          void.resize(cd.n(), T::default());
          let _ = cd.decompress(&mut void)?;
          src = cd.into_src();
          page_size += measure_bytes_read(src, prev_src_len);
        }
        _ => panic!("unreachable"),
      }
    }

    let n: usize = chunk_ns.iter().sum();
    let uncompressed_size = <T as Number>::L::BITS as usize / 8 * n;
    let compressed_size = header_size + meta_size + page_size + footer_size;
    let unknown_trailing_bytes = src.len();

    let mut chunks = Vec::new();
    for (idx, meta) in metas.iter().enumerate() {
      let latent_vars = build_latent_var_summaries::<T>(meta);
      chunks.push(ChunkSummary {
        idx,
        n: chunk_ns[idx],
        mode: format!("{:?}", meta.mode),
        delta_encoding: format!("{:?}", meta.delta_encoding),
        latent_vars,
      });
    }

    let output = Output {
      filename: opt.path.to_str().unwrap().to_string(),
      data_type: utils::dtype_name::<T>(),
      format_version: fd.format_version().to_string(),
      n,
      n_chunks: metas.len(),
      uncompressed_size,
      compressed: CompressionSummary {
        ratio: uncompressed_size as f64 / compressed_size as f64,
        total_size: compressed_size,
        header_size,
        meta_size,
        page_size,
        footer_size,
        unknown_trailing_bytes,
      },
      chunks,
    };

    println!("{}", toml::to_string_pretty(&output)?);

    Ok(())
  }
}
