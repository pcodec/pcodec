use crate::bit_writer::BitWriter;
use crate::chunk_config::DeltaSpec;
use crate::chunk_latent_compressor::{ChunkLatentCompressor, DynChunkLatentCompressor};
use crate::compression_intermediates::{BinCompressionInfo, PageInfoVar, TrainedBins};
use crate::compression_intermediates::{DissectedPage, PageInfo};
use crate::constants::{
  Bitlen, Weight, LIMITED_UNOPTIMIZED_BINS_LOG, MAX_BATCH_LATENT_VAR_SIZE, MAX_COMPRESSION_LEVEL,
  MAX_CONSECUTIVE_DELTA_ORDER, MAX_ENTRIES, OVERSHOOT_PADDING,
};
use crate::data_types::number_priv::NumberPriv;
use crate::data_types::SplitLatents;
use crate::data_types::{Latent, LatentType};
use crate::delta::DeltaState;
use crate::dyn_slices::DynNumberSlice;
use crate::errors::{PcoError, PcoResult};
use crate::histograms::histogram;
use crate::macros::{match_latent_enum, match_number_enum};
use crate::metadata::chunk_latent_var::ChunkLatentVarMeta;
use crate::metadata::dyn_bins::DynBins;
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::page::PageMeta;
use crate::metadata::page_latent_var::PageLatentVarMeta;
use crate::metadata::per_latent_var::{LatentVarKey, PerLatentVar, PerLatentVarBuilder};
use crate::metadata::{Bin, ChunkMeta, DeltaEncoding, Mode};
use crate::mode::classic;
use crate::wrapped::guarantee;
use crate::{ans, bin_optimization, delta, ChunkConfig, PagingSpec, FULL_BATCH_N};
use std::any;
use std::cmp::min;
use std::io::Write;

// if it looks like the average page of size n will use k bits, hint that it
// will be PAGE_SIZE_OVERESTIMATION * k bits.
const PAGE_SIZE_OVERESTIMATION: f64 = 1.2;
const N_PER_EXTRA_DELTA_GROUP: usize = 10000;
const DELTA_GROUP_SIZE: usize = 200;
const LOOKBACK_REQUIRED_BYTE_SAVINGS_PER_N: f32 = 0.25;

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

#[inline(never)]
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
  chunk_latent_compressors: PerLatentVar<DynChunkLatentCompressor>,
  page_infos: Vec<PageInfo>,
}

#[inline(never)]
fn bins_from_compression_infos<L: Latent>(infos: &[BinCompressionInfo<L>]) -> Vec<Bin<L>> {
  infos.iter().cloned().map(Bin::from).collect()
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

#[inline(never)]
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
  delta_encoding: &DeltaEncoding,
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
      LatentType<L> => { DynLatents::new(Vec::<L>::with_capacity(n)) }
    )
  });
  for &page_n in n_per_page {
    let end_idx = start_idx + page_n;

    let page_delta_latents = delta::compute_delta_latent_var(
      delta_encoding,
      &mut latents.primary,
      start_idx..end_idx,
    );

    let mut per_latent_var = latents.as_mut().map(|key, mut var_latents| {
      let encoding_for_var = delta_encoding.for_latent_var(key);
      let delta_state = match_latent_enum!(
        &mut var_latents,
        DynLatents<L>(var_latents) => {
          DynLatents::new(delta::encode_in_place(
            &encoding_for_var,
            page_delta_latents.as_ref(),
            &mut var_latents[start_idx..end_idx],
          ))
        }
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
          let delta_state = DeltaState::new(Vec::<L>::new());
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

#[inline(never)]
fn foo(
  key: LatentVarKey,
  unoptimized_bins_log: Bitlen,
  latents: DynLatents,
  page_infos: &[PageInfo],
) -> PcoResult<(
  ChunkLatentVarMeta,
  DynChunkLatentCompressor,
  Vec<Weight>,
)> {
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

  let res = match_latent_enum!(
    latents,
    DynLatents<L>(latents) => {
      let contiguous_deltas = collect_contiguous_latents(&latents, page_infos, key);
      let trained = train_infos(contiguous_deltas, unoptimized_bins_log)?;

      let bins = bins_from_compression_infos(&trained.infos);

      let ans_size_log = trained.ans_size_log;
      let bin_counts = trained.counts.to_vec();
      let clc = DynChunkLatentCompressor::new(
        ChunkLatentCompressor::new(trained, &bins, latents)?
      );
      let var_meta = ChunkLatentVarMeta {
        bins: DynBins::new(bins),
        ans_size_log,
      };
      (var_meta, clc, bin_counts)
    }
  );
  Ok(res)
}

fn new_candidate(
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
    delta_encode_and_build_page_infos(&delta_encoding, &n_per_page, latents);

  // training bins
  let mut var_metas = PerLatentVarBuilder::default();
  let mut chunk_latent_compressors = PerLatentVarBuilder::default();
  let mut bin_countss = PerLatentVarBuilder::default();
  for (key, latents) in latents.enumerated() {
    let (var_meta, clc, bin_counts) = foo(
      key,
      unoptimized_bins_log,
      latents,
      &page_infos,
    )?;
    var_metas.set(key, var_meta);
    chunk_latent_compressors.set(key, clc);
    bin_countss.set(key, bin_counts);
  }

  let var_metas = var_metas.into();
  let chunk_latent_compressors = chunk_latent_compressors.into();
  let bin_countss = bin_countss.into();

  let meta = ChunkMeta::new(mode, delta_encoding, var_metas)?;
  let chunk_compressor = ChunkCompressor {
    meta,
    chunk_latent_compressors,
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
      DynLatents::new(sample)
    }
  )
}

fn calculate_compressed_sample_size(
  sample: &DynLatents,
  unoptimized_bins_log: Bitlen,
  delta_encoding: DeltaEncoding,
) -> PcoResult<f32> {
  let sample_n = sample.len();
  let (sample_cc, _) = new_candidate(
    SplitLatents {
      primary: sample.clone(),
      secondary: None,
    },
    &PagingSpec::Exact(vec![sample_n]),
    Mode::Classic,
    delta_encoding,
    unoptimized_bins_log,
  )?;
  let size = sample_cc.meta_size_hint() + sample_cc.page_size_hint_inner(0, 1.0);
  Ok(size as f32)
}

#[inline(never)]
fn choose_auto_delta_encoding(
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

  let mut best_encoding = DeltaEncoding::NoOp;
  let mut best_cost = calculate_compressed_sample_size(
    &sample,
    unoptimized_bins_log,
    DeltaEncoding::NoOp,
  )?;

  let lookback_penalty = LOOKBACK_REQUIRED_BYTE_SAVINGS_PER_N * sample_n as f32;
  if best_cost > lookback_penalty {
    let lookback_encoding = delta::new_lookback(sample_n);
    let lookback_cost = calculate_compressed_sample_size(
      &sample,
      unoptimized_bins_log,
      lookback_encoding.clone(),
    )? + lookback_penalty;
    if lookback_cost < best_cost {
      best_encoding = delta::new_lookback(primary_latents.len());
      best_cost = lookback_cost;
    }
  }

  for delta_encoding_order in 1..MAX_CONSECUTIVE_DELTA_ORDER + 1 {
    let encoding = DeltaEncoding::Consecutive {
      order: delta_encoding_order,
      secondary_uses_delta: false,
    };
    let cost = calculate_compressed_sample_size(
      &sample,
      unoptimized_bins_log,
      encoding.clone(),
    )?;
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

fn choose_delta_encoding(
  latents: &SplitLatents,
  config: &ChunkConfig,
  unoptimized_bins_log: Bitlen,
) -> PcoResult<DeltaEncoding> {
  let n = latents.primary.len();
  let delta_encoding = match config.delta_spec {
    DeltaSpec::Auto => choose_auto_delta_encoding(&latents.primary, unoptimized_bins_log)?,
    DeltaSpec::NoOp | DeltaSpec::TryConsecutive(0) | DeltaSpec::TryConv1(0) => DeltaEncoding::NoOp,
    DeltaSpec::TryConsecutive(order) => DeltaEncoding::Consecutive {
      order,
      secondary_uses_delta: false,
    },
    DeltaSpec::TryLookback => delta::new_lookback(n),
    DeltaSpec::TryConv1(order) => {
      // We use the entire chunk to fit the weights and bias here. In the
      // future, it may be ideal to take into account the page boundaries.
      delta::new_conv1(order, &latents.primary)?.unwrap_or(DeltaEncoding::NoOp)
    }
  };
  Ok(delta_encoding)
}

fn fallback_chunk_compressor(
  latents: SplitLatents,
  config: &ChunkConfig,
) -> PcoResult<ChunkCompressor> {
  let n = latents.primary.len();
  let n_per_page = config.paging_spec.n_per_page(n)?;
  let (latents, page_infos) =
    delta_encode_and_build_page_infos(&DeltaEncoding::NoOp, &n_per_page, latents);

  let (meta, clc) = match_latent_enum!(
    latents.primary,
    DynLatents<L>(latents) => {
      let infos = vec![BinCompressionInfo::<L> {
        weight: 1,
        symbol: 0,
        ..Default::default()
      }];
      let meta = guarantee::baseline_chunk_meta::<L>();
      let latent_var_meta = &meta.per_latent_var.primary;

      let clc = ChunkLatentCompressor::new(
        TrainedBins {
          infos,
          ans_size_log: 0,
          counts: vec![n as Weight],
        },
        latent_var_meta.bins.downcast_ref::<L>().unwrap(),
        latents,
      )?;
      (meta, DynChunkLatentCompressor::new(clc))
    }
  );

  Ok(ChunkCompressor {
    meta,
    chunk_latent_compressors: PerLatentVar {
      delta: None,
      primary: clc,
      secondary: None,
    },
    page_infos,
  })
}

impl ChunkCompressor {
  // This is where the bulk of compression happens.
  pub(crate) fn new(nums: DynNumberSlice, config: &ChunkConfig) -> PcoResult<Self> {
    let latent_type = match_number_enum!(
      &nums,
      DynNumberSlice<T>(_inner) => {
        LatentType::new::<<T as NumberPriv>::L>()
      }
    );
    config.validate(latent_type)?;
    let n = nums.len();
    validate_chunk_size(n)?;

    // 1. choose mode and split the latents
    // TODO in a later PR: validate mode on initialization of Mode or maybe ChunkMeta
    let (mode, latents) = match_number_enum!(
      nums,
      DynNumberSlice<T>(nums) => {
        let (mode, latents) = T::choose_mode_and_split_latents(nums, config)?;
        if !T::mode_is_valid(&mode) {
          return Err(PcoError::invalid_argument(format!(
            "The chosen mode of {:?} was invalid for type {}. \
            This is most likely due to an invalid argument, but if using Auto mode \
            spec, it could also be a bug in pco.",
            mode,
            any::type_name::<T>(),
          )));
        }
        (mode, latents)
      }
    );

    // 2. choose delta encoding
    let unoptimized_bins_log = choose_unoptimized_bins_log(config.compression_level, n);
    let delta_encoding = choose_delta_encoding(&latents, config, unoptimized_bins_log)?;

    // 3. apply the delta encoding and choose bins
    // These steps are together because it's convenient; they both do logic per
    // page and in a latent-type-specialized way.
    let (candidate, bin_counts) = new_candidate(
      latents,
      &config.paging_spec,
      mode,
      delta_encoding,
      unoptimized_bins_log,
    )?;

    // 4. check that our compressed size meets guarantees and fall back if not
    if candidate.should_fallback(latent_type, n, bin_counts) {
      let split_latents = match_number_enum!(
        nums,
        DynNumberSlice<T>(nums) => {
          classic::split_latents(nums)
        }
      );
      return fallback_chunk_compressor(split_latents, config);
    }

    Ok(candidate)
  }

  fn should_fallback(
    &self,
    latent_type: LatentType,
    n: usize,
    bin_counts_per_latent_var: PerLatentVar<Vec<Weight>>,
  ) -> bool {
    let meta = &self.meta;
    if matches!(meta.delta_encoding, DeltaEncoding::NoOp) && matches!(meta.mode, Mode::Classic) {
      // we already have a size guarantee in this case
      return false;
    }

    let n_pages = self.page_infos.len();

    // worst case trailing bits after bit packing
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

    let worst_case_size = meta.max_size()
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
  pub fn meta_size_hint(&self) -> usize {
    self.meta.max_size()
  }

  /// Writes the chunk metadata to the destination.
  ///
  /// Will return an error if the provided `Write` errors.
  pub fn write_meta<W: Write>(&self, dst: W) -> PcoResult<W> {
    let mut writer = BitWriter::new(dst, self.meta.max_size() + OVERSHOOT_PADDING);
    unsafe { self.meta.write_to(&mut writer)? };
    Ok(writer.into_inner())
  }

  fn dissect_page(&mut self, page_idx: usize) -> PcoResult<DissectedPage> {
    let Self {
      chunk_latent_compressors,
      page_infos,
      ..
    } = self;

    let page_info = &page_infos[page_idx];

    let per_latent_var = chunk_latent_compressors.as_mut().map(|key, clc| {
      let range = page_info.range_for_latent_var(key);
      match_latent_enum!(
        clc,
        DynChunkLatentCompressor<L>(inner) => {
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
  /// you can `.reserve(chunk_compressor.size_hint())` ahead of time.
  pub fn page_size_hint(&self, page_idx: usize) -> usize {
    self.page_size_hint_inner(page_idx, PAGE_SIZE_OVERESTIMATION)
  }

  fn page_size_hint_inner(&self, page_idx: usize, page_size_overestimation: f64) -> usize {
    let page_info = &self.page_infos[page_idx];
    let mut body_bit_size = 0;
    for (_, (clc, page_info_var)) in self
      .chunk_latent_compressors
      .as_ref()
      .zip_exact(page_info.per_latent_var.as_ref())
      .enumerated()
    {
      let n_stored_latents = page_info_var.range.len();
      let avg_bits_per_latent = match_latent_enum!(
        clc,
        DynChunkLatentCompressor<L>(inner) => { inner.avg_bits_per_latent }
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
      for (_, (page_dissected_var, clc)) in dissected_page
        .per_latent_var
        .as_ref()
        .zip_exact(self.chunk_latent_compressors.as_ref())
        .enumerated()
      {
        match_latent_enum!(
          clc,
          DynChunkLatentCompressor<L>(inner) => {
            inner.write_dissected_batch(page_dissected_var, batch_start, writer)?;
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
  ///
  /// Even though this takes `&mut self`, it only mutates scratch buffers and
  /// has no effect on the compression of later pages.
  pub fn write_page<W: Write>(&mut self, page_idx: usize, dst: W) -> PcoResult<W> {
    let n_pages = self.page_infos.len();
    if page_idx >= n_pages {
      return Err(PcoError::invalid_argument(format!(
        "page idx exceeds num pages ({} >= {})",
        page_idx, n_pages,
      )));
    }

    let mut writer = BitWriter::new(dst, MAX_BATCH_LATENT_VAR_SIZE);

    let dissected_page = self.dissect_page(page_idx)?;
    let page_info = &self.page_infos[page_idx];

    let ans_default_state_and_size_log = self.chunk_latent_compressors.as_ref().map(|_, clc| {
      match_latent_enum!(
        clc,
        DynChunkLatentCompressor<L>(inner) => { (inner.encoder.default_state(), inner.encoder.size_log()) }
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
    let latents = DynLatents::new(vec![0_u32, 1]);
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

    let latents = DynLatents::new((0..300).collect::<Vec<u32>>());
    let sample = choose_delta_sample(&latents, 100, 1)
      .downcast::<u32>()
      .unwrap();
    assert_eq!(sample.len(), 200);
    assert_eq!(&sample[..3], &[0, 1, 2]);
    assert_eq!(&sample[197..], &[297, 298, 299]);

    let latents = DynLatents::new((0..8).collect::<Vec<u32>>());
    assert_eq!(
      choose_delta_sample(&latents, 2, 2)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1, 3, 4, 6, 7]
    );
  }
}
