use crate::bit_writer::BitWriter;
use crate::chunk_config::DeltaSpec;
use crate::compression_intermediates::{BinCompressionInfo, PageInfoVar};
use crate::compression_intermediates::{DissectedPage, PageInfo};
use crate::constants::{
  Bitlen, Weight, LIMITED_UNOPTIMIZED_BINS_LOG, MAX_COMPRESSION_LEVEL, MAX_DELTA_ENCODING_ORDER,
  MAX_ENTRIES, OVERSHOOT_PADDING, PAGE_PADDING,
};
use crate::data_types::SplitLatents;
use crate::data_types::{Latent, LatentType, Number};
use crate::delta::DeltaState;
use crate::errors::{PcoError, PcoResult};
use crate::histograms::histogram;
use crate::latent_chunk_compressor::{
  DynLatentChunkCompressor, LatentChunkCompressor, TrainedBins,
};
use crate::macros::match_latent_enum;
use crate::metadata::chunk_latent_var::ChunkLatentVarMeta;
use crate::metadata::delta_encoding::{DeltaConsecutiveConfig, DeltaLookbackConfig};
use crate::metadata::dyn_bins::DynBins;
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::page::PageMeta;
use crate::metadata::page_latent_var::PageLatentVarMeta;
use crate::metadata::per_latent_var::{LatentVarKey, PerLatentVar, PerLatentVarBuilder};
use crate::metadata::{Bin, ChunkMeta, DeltaEncoding, Mode};
use crate::wrapped::guarantee;
use crate::{
  ans, bin_optimization, bits, data_types, delta, ChunkConfig, PagingSpec, FULL_BATCH_N,
};
use std::cmp::min;
use std::io::Write;

// if it looks like the average page of size n will use k bits, hint that it
// will be PAGE_SIZE_OVERESTIMATION * k bits.
const PAGE_SIZE_OVERESTIMATION: f64 = 1.2;
const N_PER_EXTRA_DELTA_GROUP: usize = 10000;
const DELTA_GROUP_SIZE: usize = 200;
const LOOKBACK_MAX_WINDOW_N_LOG: Bitlen = 15;
const LOOKBACK_MIN_WINDOW_N_LOG: Bitlen = 4;
const LOOKBACK_REQUIRED_BYTE_SAVINGS_PER_N: f32 = 0.25;

// TODO taking deltas of secondary latents has been proven to help slightly
// in some cases, so we should consider it in the future

fn new_lookback_delta_encoding(n: usize) -> DeltaEncoding {
  DeltaEncoding::Lookback(DeltaLookbackConfig {
    window_n_log: bits::bits_to_encode_offset(n as u32 - 1).clamp(
      LOOKBACK_MIN_WINDOW_N_LOG,
      LOOKBACK_MAX_WINDOW_N_LOG,
    ),
    state_n_log: 0,
    secondary_uses_delta: false,
  })
}

// returns table size log
fn quantize_weights<L: Latent>(
  infos: &mut [BinCompressionInfo<L>],
  n_latents: usize,
  estimated_ans_size_log: Bitlen,
) -> Bitlen {
  let counts = infos.iter().map(|info| info.weight).collect::<Vec<_>>();
  let (ans_size_log, weights) = ans::quantize_weights(counts, n_latents, estimated_ans_size_log);

  for (i, weight) in weights.into_iter().enumerate() {
    infos[i].weight = weight;
  }
  ans_size_log
}

fn train_infos<L: Latent>(
  mut latents: Vec<L>,
  unoptimized_bins_log: Bitlen,
) -> PcoResult<TrainedBins<L>> {
  if latents.is_empty() {
    return Ok(TrainedBins::default());
  }

  let n_latents = latents.len();
  let unoptimized_bins = histogram(&mut latents, unoptimized_bins_log as Bitlen);

  let n_log_ceil = if n_latents <= 1 {
    0
  } else {
    (n_latents - 1).ilog2() + 1
  };
  // We cap the ANS table size so that it fits into L1 (or at least L2) cache
  // and has predictably small bitlengths for fast decompression.
  // Maybe in the future we could extend this to MAX_ANS_BITS (14) if the user
  // enables something. We should definitely quantize more aggressively if we
  // do that.
  let estimated_ans_size_log = min(
    min(
      (unoptimized_bins_log + 2) as Bitlen,
      MAX_COMPRESSION_LEVEL as Bitlen,
    ),
    n_log_ceil,
  );

  let mut optimized_infos =
    bin_optimization::optimize_bins(&unoptimized_bins, estimated_ans_size_log);

  let counts = optimized_infos
    .iter()
    .map(|info| info.weight)
    .collect::<Vec<_>>();
  let ans_size_log = quantize_weights(
    &mut optimized_infos,
    n_latents,
    estimated_ans_size_log,
  );

  Ok(TrainedBins {
    infos: optimized_infos,
    ans_size_log,
    counts,
  })
}

/// Holds metadata about a chunk and supports compression.
#[derive(Clone, Debug)]
pub struct ChunkCompressor {
  meta: ChunkMeta,
  latent_chunk_compressors: PerLatentVar<DynLatentChunkCompressor>,
  page_infos: Vec<PageInfo>,
}

fn bins_from_compression_infos<L: Latent>(infos: &[BinCompressionInfo<L>]) -> Vec<Bin<L>> {
  infos.iter().cloned().map(Bin::from).collect()
}

fn validate_config(config: &ChunkConfig) -> PcoResult<()> {
  let compression_level = config.compression_level;
  if compression_level > MAX_COMPRESSION_LEVEL {
    return Err(PcoError::invalid_argument(format!(
      "compression level may not exceed {} (was {})",
      MAX_COMPRESSION_LEVEL, compression_level,
    )));
  }

  if let DeltaSpec::TryConsecutive(order) = config.delta_spec {
    if order > MAX_DELTA_ENCODING_ORDER {
      return Err(PcoError::invalid_argument(format!(
        "delta encoding order may not exceed {} (was {})",
        MAX_DELTA_ENCODING_ORDER, order,
      )));
    }
  }

  Ok(())
}

fn validate_chunk_size(n: usize) -> PcoResult<()> {
  if n == 0 {
    return Err(PcoError::invalid_argument(
      "cannot compress empty chunk",
    ));
  }
  if n > MAX_ENTRIES {
    return Err(PcoError::invalid_argument(format!(
      "count may not exceed {} per chunk (was {})",
      MAX_ENTRIES, n,
    )));
  }

  Ok(())
}

fn collect_contiguous_latents<L: Latent>(
  latents: &[L],
  page_infos: &[PageInfo],
  latent_var_key: LatentVarKey,
) -> Vec<L> {
  let mut res = Vec::with_capacity(latents.len());
  for page in page_infos {
    let range = page.range_for_latent_var(latent_var_key);
    res.extend(&latents[range]);
  }
  res
}

fn delta_encode_and_build_page_infos(
  delta_encoding: DeltaEncoding,
  n_per_page: &[usize],
  latents: SplitLatents,
) -> (PerLatentVar<DynLatents>, Vec<PageInfo>) {
  let n = latents.primary.len();
  let mut latents = PerLatentVar {
    delta: None,
    primary: latents.primary,
    secondary: latents.secondary,
  };
  let n_pages = n_per_page.len();
  let mut page_infos = Vec::with_capacity(n_pages);

  // delta encoding
  let mut start_idx = 0;
  let mut delta_latents = delta_encoding.latent_type().map(|ltype| {
    match_latent_enum!(
      ltype,
      LatentType<L> => { DynLatents::new(Vec::<L>::with_capacity(n)).unwrap() }
    )
  });
  for &page_n in n_per_page {
    let end_idx = start_idx + page_n;

    let page_delta_latents = delta::compute_delta_latent_var(
      delta_encoding,
      &mut latents.primary,
      start_idx..end_idx,
    );

    let mut per_latent_var = latents.as_mut().map(|key, var_latents| {
      let encoding_for_var = delta_encoding.for_latent_var(key);
      let delta_state = delta::encode_in_place(
        encoding_for_var,
        page_delta_latents.as_ref(),
        start_idx..end_idx,
        var_latents,
      );
      // delta encoding in place leaves junk in the first n_latents_per_state
      let stored_start_idx = min(
        start_idx + encoding_for_var.n_latents_per_state(),
        end_idx,
      );
      let range = stored_start_idx..end_idx;
      PageInfoVar { delta_state, range }
    });

    if let Some(delta_latents) = delta_latents.as_mut() {
      match_latent_enum!(
        delta_latents,
        DynLatents<L>(delta_latents) => {
          let page_delta_latents = page_delta_latents.unwrap().downcast::<L>().unwrap();
          let delta_state = DeltaState::new(Vec::<L>::new()).unwrap();
          let range = delta_latents.len()..delta_latents.len() + page_delta_latents.len();
          per_latent_var.delta = Some(PageInfoVar { delta_state, range });
          delta_latents.extend(&page_delta_latents);
        }
      )
    }

    page_infos.push(PageInfo {
      page_n,
      per_latent_var,
    });

    start_idx = end_idx;
  }
  latents.delta = delta_latents;

  (latents, page_infos)
}

fn new_candidate_w_split_and_delta_encoding(
  latents: SplitLatents, // start out plain, gets delta encoded in place
  paging_spec: &PagingSpec,
  mode: Mode,
  delta_encoding: DeltaEncoding,
  unoptimized_bins_log: Bitlen,
) -> PcoResult<(ChunkCompressor, PerLatentVar<Vec<Weight>>)> {
  let chunk_n = latents.primary.len();
  let n_per_page = paging_spec.n_per_page(chunk_n)?;

  // delta encoding
  let (latents, page_infos) =
    delta_encode_and_build_page_infos(delta_encoding, &n_per_page, latents);

  // training bins
  let mut var_metas = PerLatentVarBuilder::default();
  let mut latent_chunk_compressors = PerLatentVarBuilder::default();
  let mut bin_countss = PerLatentVarBuilder::default();
  for (key, latents) in latents.enumerated() {
    let unoptimized_bins_log = match key {
      // primary latents are generally the most important to compress, and
      // delta latents typically have a small number of discrete values, so
      // aren't slow to optimize anyway
      LatentVarKey::Delta | LatentVarKey::Primary => unoptimized_bins_log,
      // secondary latents should be compressed faster
      LatentVarKey::Secondary => min(
        unoptimized_bins_log,
        LIMITED_UNOPTIMIZED_BINS_LOG,
      ),
    };

    let (var_meta, lcc, bin_counts) = match_latent_enum!(
      latents,
      DynLatents<L>(latents) => {
        let contiguous_deltas = collect_contiguous_latents(&latents, &page_infos, key);
        let trained = train_infos(contiguous_deltas, unoptimized_bins_log)?;

        let bins = bins_from_compression_infos(&trained.infos);

        let ans_size_log = trained.ans_size_log;
        let bin_counts = trained.counts.to_vec();
        let lcc = DynLatentChunkCompressor::new(
          LatentChunkCompressor::new(trained, &bins, latents)?
        ).unwrap();
        let var_meta = ChunkLatentVarMeta {
          bins: DynBins::new(bins).unwrap(),
          ans_size_log,
        };
        (var_meta, lcc, bin_counts)
      }
    );
    var_metas.set(key, var_meta);
    latent_chunk_compressors.set(key, lcc);
    bin_countss.set(key, bin_counts);
  }

  let var_metas = var_metas.into();
  let latent_chunk_compressors = latent_chunk_compressors.into();
  let bin_countss = bin_countss.into();

  let meta = ChunkMeta {
    mode,
    delta_encoding,
    per_latent_var: var_metas,
  };
  let chunk_compressor = ChunkCompressor {
    meta,
    latent_chunk_compressors,
    page_infos,
  };

  Ok((chunk_compressor, bin_countss))
}

fn choose_delta_sample(
  primary_latents: &DynLatents,
  group_size: usize,
  n_extra_groups: usize,
) -> DynLatents {
  let n = primary_latents.len();
  let nominal_sample_size = (n_extra_groups + 1) * group_size;
  let group_padding = if n_extra_groups == 0 {
    0
  } else {
    n.saturating_sub(nominal_sample_size) / n_extra_groups
  };

  let mut i = group_size;

  match_latent_enum!(
    primary_latents,
    DynLatents<L>(primary_latents) => {
      let mut sample = Vec::<L>::with_capacity(nominal_sample_size);
      sample.extend(primary_latents.iter().take(group_size));
      for _ in 0..n_extra_groups {
        i += group_padding;
        sample.extend(primary_latents.iter().skip(i).take(group_size));
        i += group_size;
      }
      DynLatents::new(sample).unwrap()
    }
  )
}

fn calculate_compressed_sample_size(
  sample: &DynLatents,
  unoptimized_bins_log: Bitlen,
  delta_encoding: DeltaEncoding,
) -> PcoResult<f32> {
  let sample_n = sample.len();
  let (sample_cc, _) = new_candidate_w_split_and_delta_encoding(
    SplitLatents {
      primary: sample.clone(),
      secondary: None,
    },
    &PagingSpec::Exact(vec![sample_n]),
    Mode::Classic,
    delta_encoding,
    unoptimized_bins_log,
  )?;
  let size = sample_cc.chunk_meta_size_hint() + sample_cc.page_size_hint_inner(0, 1.0);
  Ok(size as f32)
}

#[inline(never)]
fn choose_delta_encoding(
  primary_latents: &DynLatents,
  unoptimized_bins_log: Bitlen,
) -> PcoResult<DeltaEncoding> {
  let n = primary_latents.len();
  let sample = choose_delta_sample(
    primary_latents,
    DELTA_GROUP_SIZE,
    1 + n / N_PER_EXTRA_DELTA_GROUP,
  );
  let sample_n = sample.len();

  let mut best_encoding = DeltaEncoding::None;
  let mut best_cost = calculate_compressed_sample_size(
    &sample,
    unoptimized_bins_log,
    DeltaEncoding::None,
  )?;

  let lookback_penalty = LOOKBACK_REQUIRED_BYTE_SAVINGS_PER_N * sample_n as f32;
  if best_cost > lookback_penalty {
    let lookback_encoding = new_lookback_delta_encoding(sample_n);
    let lookback_cost = calculate_compressed_sample_size(
      &sample,
      unoptimized_bins_log,
      lookback_encoding,
    )? + lookback_penalty;
    if lookback_cost < best_cost {
      best_encoding = new_lookback_delta_encoding(primary_latents.len());
      best_cost = lookback_cost;
    }
  }

  for delta_encoding_order in 1..MAX_DELTA_ENCODING_ORDER + 1 {
    let encoding = DeltaEncoding::Consecutive(DeltaConsecutiveConfig {
      order: delta_encoding_order,
      secondary_uses_delta: false,
    });
    let cost = calculate_compressed_sample_size(&sample, unoptimized_bins_log, encoding)?;
    if cost < best_cost {
      best_encoding = encoding;
      best_cost = cost;
    } else {
      // it's almost always convex
      break;
    }
  }

  Ok(best_encoding)
}

fn choose_unoptimized_bins_log(compression_level: usize, n: usize) -> Bitlen {
  let compression_level = compression_level as Bitlen;
  let log_n = (n as f64).log2().floor() as Bitlen;
  let fast_unoptimized_bins_log = log_n.saturating_sub(4);
  if compression_level <= fast_unoptimized_bins_log {
    compression_level
  } else {
    fast_unoptimized_bins_log + compression_level.saturating_sub(fast_unoptimized_bins_log) / 2
  }
}

// We pull this stuff out of `new` because it only depends on the latent type
// and we don't need a specialization for each full number type.
// Returns a chunk compressor and the counts (per latent var) of numbers in
// each bin.
fn new_candidate_w_split(
  mode: Mode,
  latents: SplitLatents,
  config: &ChunkConfig,
) -> PcoResult<(ChunkCompressor, PerLatentVar<Vec<Weight>>)> {
  let n = latents.primary.len();
  let unoptimized_bins_log = choose_unoptimized_bins_log(config.compression_level, n);
  let delta_encoding = match config.delta_spec {
    DeltaSpec::Auto => choose_delta_encoding(&latents.primary, unoptimized_bins_log)?,
    DeltaSpec::None | DeltaSpec::TryConsecutive(0) => DeltaEncoding::None,
    DeltaSpec::TryConsecutive(order) => DeltaEncoding::Consecutive(DeltaConsecutiveConfig {
      order,
      secondary_uses_delta: false,
    }),
    DeltaSpec::TryLookback => new_lookback_delta_encoding(n),
  };

  new_candidate_w_split_and_delta_encoding(
    latents,
    &config.paging_spec,
    mode,
    delta_encoding,
    unoptimized_bins_log,
  )
}

fn fallback_chunk_compressor(
  latents: SplitLatents,
  config: &ChunkConfig,
) -> PcoResult<ChunkCompressor> {
  let n = latents.primary.len();
  let n_per_page = config.paging_spec.n_per_page(n)?;
  let (latents, page_infos) =
    delta_encode_and_build_page_infos(DeltaEncoding::None, &n_per_page, latents);

  let (meta, lcc) = match_latent_enum!(
    latents.primary,
    DynLatents<L>(latents) => {
      let infos = vec![BinCompressionInfo::<L> {
        weight: 1,
        symbol: 0,
        ..Default::default()
      }];
      let meta = guarantee::baseline_chunk_meta::<L>();
      let latent_var_meta = &meta.per_latent_var.primary;

      let lcc = LatentChunkCompressor::new(
        TrainedBins {
          infos,
          ans_size_log: 0,
          counts: vec![n as Weight],
        },
        latent_var_meta.bins.downcast_ref::<L>().unwrap(),
        latents,
      )?;
      (meta, DynLatentChunkCompressor::new(lcc).unwrap())
    }
  );

  Ok(ChunkCompressor {
    meta,
    latent_chunk_compressors: PerLatentVar {
      delta: None,
      primary: lcc,
      secondary: None,
    },
    page_infos,
  })
}

// Should this take nums as a slice of slices instead of having a config.paging_spec?
pub(crate) fn new<T: Number>(nums: &[T], config: &ChunkConfig) -> PcoResult<ChunkCompressor> {
  validate_config(config)?;
  let n = nums.len();
  validate_chunk_size(n)?;

  let (mode, latents) = T::choose_mode_and_split_latents(nums, config)?;
  if !T::mode_is_valid(mode) {
    return Err(PcoError::invalid_argument(
      "The chosen mode of {:?} was invalid for type {}. \
      This is most likely due to an invalid argument, but if using Auto mode \
      spec, it could also be a bug in pco.",
    ));
  }

  let (candidate, bin_counts) = new_candidate_w_split(mode, latents, config)?;
  if candidate.should_fallback(
    LatentType::new::<T::L>().unwrap(),
    n,
    bin_counts,
  ) {
    let split_latents = data_types::split_latents_classic(nums);
    return fallback_chunk_compressor(split_latents, config);
  }

  Ok(candidate)
}

impl ChunkCompressor {
  fn should_fallback(
    &self,
    latent_type: LatentType,
    n: usize,
    bin_counts_per_latent_var: PerLatentVar<Vec<Weight>>,
  ) -> bool {
    let meta = &self.meta;
    if meta.delta_encoding == DeltaEncoding::None && meta.mode == Mode::Classic {
      // we already have a size guarantee in this case
      return false;
    }

    let n_pages = self.page_infos.len();

    // worst case trailing bytes after bit packing
    let mut worst_case_body_bit_size = 7 * n_pages;
    for (_, (latent_var_meta, bin_counts)) in meta
      .per_latent_var
      .as_ref()
      .zip_exact(bin_counts_per_latent_var.as_ref())
      .enumerated()
    {
      match_latent_enum!(&latent_var_meta.bins, DynBins<L>(bins) => {
        for (bin, &count) in bins.iter().zip(bin_counts) {
          worst_case_body_bit_size +=
            count as usize * bin.worst_case_bits_per_latent(latent_var_meta.ans_size_log) as usize;
        }
      });
    }

    let worst_case_size = meta.exact_size()
      + n_pages * meta.exact_page_meta_size()
      + worst_case_body_bit_size.div_ceil(8);

    let baseline_size = match_latent_enum!(
      latent_type,
      LatentType<L> => { guarantee::chunk_size::<L>(n) }
    );
    worst_case_size > baseline_size
  }

  /// Returns the count of numbers this chunk will contain in each page.
  pub fn n_per_page(&self) -> Vec<usize> {
    self.page_infos.iter().map(|page| page.page_n).collect()
  }

  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.meta
  }

  /// Returns an estimate of the overall size of the chunk.
  ///
  /// This can be useful when building the file as a `Vec<u8>` in memory;
  /// you can `.reserve()` ahead of time.
  pub fn chunk_meta_size_hint(&self) -> usize {
    self.meta.exact_size()
  }

  /// Writes the chunk metadata to the destination.
  ///
  /// Will return an error if the provided `Write` errors.
  pub fn write_chunk_meta<W: Write>(&self, dst: W) -> PcoResult<W> {
    let mut writer = BitWriter::new(
      dst,
      self.meta.exact_size() + OVERSHOOT_PADDING,
    );
    unsafe { self.meta.write_to(&mut writer)? };
    Ok(writer.into_inner())
  }

  fn dissect_page(&self, page_idx: usize) -> PcoResult<DissectedPage> {
    let Self {
      latent_chunk_compressors,
      page_infos,
      ..
    } = self;

    let page_info = &page_infos[page_idx];

    let per_latent_var = latent_chunk_compressors.as_ref().map(|key, lcc| {
      let range = page_info.range_for_latent_var(key);
      match_latent_enum!(
        lcc,
        DynLatentChunkCompressor<L>(inner) => {
          inner.dissect_page(range)
        }
      )
    });

    Ok(DissectedPage {
      page_n: page_info.page_n,
      per_latent_var,
    })
  }

  /// Returns an estimate of the overall size of a specific page.
  ///
  /// This can be useful when building the file as a `Vec<u8>` in memory;
  /// you can `.reserve(chunk_compressor.chunk_size_hint())` ahead of time.
  pub fn page_size_hint(&self, page_idx: usize) -> usize {
    self.page_size_hint_inner(page_idx, PAGE_SIZE_OVERESTIMATION)
  }

  fn page_size_hint_inner(&self, page_idx: usize, page_size_overestimation: f64) -> usize {
    let page_info = &self.page_infos[page_idx];
    let mut body_bit_size = 0;
    for (_, (lcc, page_info_var)) in self
      .latent_chunk_compressors
      .as_ref()
      .zip_exact(page_info.per_latent_var.as_ref())
      .enumerated()
    {
      let n_stored_latents = page_info_var.range.len();
      let avg_bits_per_latent = match_latent_enum!(
        lcc,
        DynLatentChunkCompressor<L>(inner) => { inner.avg_bits_per_latent }
      );
      let nums_bit_size = n_stored_latents as f64 * avg_bits_per_latent;
      body_bit_size += (nums_bit_size * page_size_overestimation).ceil() as usize;
    }
    self.meta.exact_page_meta_size() + body_bit_size.div_ceil(8)
  }

  #[inline(never)]
  fn write_dissected_page<W: Write>(
    &self,
    dissected_page: DissectedPage,
    writer: &mut BitWriter<W>,
  ) -> PcoResult<()> {
    let mut batch_start = 0;
    while batch_start < dissected_page.page_n {
      let batch_end = min(
        batch_start + FULL_BATCH_N,
        dissected_page.page_n,
      );
      for (_, (dissected_page_var, lcc)) in dissected_page
        .per_latent_var
        .as_ref()
        .zip_exact(self.latent_chunk_compressors.as_ref())
        .enumerated()
      {
        match_latent_enum!(
          lcc,
          DynLatentChunkCompressor<L>(inner) => {
            inner.write_dissected_batch(dissected_page_var, batch_start, writer)?;
          }
        );
      }
      batch_start = batch_end;
    }
    Ok(())
  }

  /// Writes a page to the destination.
  ///
  /// Will return an error if the provided `Write` errors.
  pub fn write_page<W: Write>(&self, page_idx: usize, dst: W) -> PcoResult<W> {
    let n_pages = self.page_infos.len();
    if page_idx >= n_pages {
      return Err(PcoError::invalid_argument(format!(
        "page idx exceeds num pages ({} >= {})",
        page_idx, n_pages,
      )));
    }

    let mut writer = BitWriter::new(dst, PAGE_PADDING);

    let dissected_page = self.dissect_page(page_idx)?;
    let page_info = &self.page_infos[page_idx];

    let ans_default_state_and_size_log = self.latent_chunk_compressors.as_ref().map(|_, lcc| {
      match_latent_enum!(
        lcc,
        DynLatentChunkCompressor<L>(inner) => { (inner.encoder.default_state(), inner.encoder.size_log()) }
      )
    });

    let per_latent_var = page_info
      .per_latent_var
      .as_ref()
      .zip_exact(ans_default_state_and_size_log.as_ref())
      .zip_exact(dissected_page.per_latent_var.as_ref())
      .map(|_, tuple| {
        let ((page_info_var, (ans_default_state, _)), dissected) = tuple;
        let ans_final_state_idxs = dissected
          .ans_final_states
          .map(|state| state - ans_default_state);
        PageLatentVarMeta {
          delta_state: page_info_var.delta_state.clone(),
          ans_final_state_idxs,
        }
      });

    let page_meta = PageMeta { per_latent_var };
    let ans_size_logs = ans_default_state_and_size_log.map(|_, (_, size_log)| size_log);
    unsafe { page_meta.write_to(ans_size_logs, &mut writer) };

    self.write_dissected_page(dissected_page, &mut writer)?;

    writer.finish_byte();
    writer.flush()?;
    Ok(writer.into_inner())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_choose_delta_sample() {
    let latents = DynLatents::new(vec![0_u32, 1]).unwrap();
    assert_eq!(
      choose_delta_sample(&latents, 100, 0)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );
    assert_eq!(
      choose_delta_sample(&latents, 100, 1)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );

    let latents = DynLatents::new((0..300).collect::<Vec<u32>>()).unwrap();
    let sample = choose_delta_sample(&latents, 100, 1)
      .downcast::<u32>()
      .unwrap();
    assert_eq!(sample.len(), 200);
    assert_eq!(&sample[..3], &[0, 1, 2]);
    assert_eq!(&sample[197..], &[297, 298, 299]);

    let latents = DynLatents::new((0..8).collect::<Vec<u32>>()).unwrap();
    assert_eq!(
      choose_delta_sample(&latents, 2, 2)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1, 3, 4, 6, 7]
    );
  }
}
