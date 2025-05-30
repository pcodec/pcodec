use std::cmp::min;
use std::fmt::Debug;
use std::marker::PhantomData;

use better_io::BetterBufRead;

use crate::bit_reader;
use crate::bit_reader::BitReaderBuilder;
use crate::constants::{FULL_BATCH_N, PAGE_PADDING};
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::latent_page_decompressor::DynLatentPageDecompressor;
use crate::macros::match_latent_enum;
use crate::metadata::page::PageMeta;
use crate::metadata::per_latent_var::{PerLatentVar, PerLatentVarBuilder};
use crate::metadata::{ChunkMeta, DeltaEncoding, DynBins, DynLatents, Mode};
use crate::progress::Progress;

const PERFORMANT_BUF_READ_CAPACITY: usize = 8192;

#[derive(Debug)]
struct LatentScratch {
  is_constant: bool,
  dst: DynLatents,
}

struct PageDecompressorInner<R: BetterBufRead> {
  // immutable
  n: usize,
  mode: Mode,
  delta_encoding: DeltaEncoding,

  // mutable
  reader_builder: BitReaderBuilder<R>,
  n_processed: usize,
  latent_decompressors: PerLatentVar<DynLatentPageDecompressor>,
  delta_scratch: Option<LatentScratch>,
  secondary_scratch: Option<LatentScratch>,
}

/// Holds metadata about a page and supports decompression.
pub struct PageDecompressor<T: Number, R: BetterBufRead> {
  inner: PageDecompressorInner<R>,
  phantom: PhantomData<T>,
}

fn convert_from_latents_to_numbers<T: Number>(dst: &mut [T]) {
  // we wrote the joined latents to dst, so we can convert them in place
  for l_and_dst in dst {
    *l_and_dst = T::from_latent_ordered(l_and_dst.transmute_to_latent());
  }
}

fn make_latent_scratch(lpd: Option<&DynLatentPageDecompressor>) -> Option<LatentScratch> {
  let lpd = lpd?;

  match_latent_enum!(
    lpd,
    DynLatentPageDecompressor<L>(inner) => {
      let maybe_constant_value = inner.maybe_constant_value;
      Some(LatentScratch {
        is_constant: maybe_constant_value.is_some(),
        dst: DynLatents::new(vec![maybe_constant_value.unwrap_or_default(); FULL_BATCH_N]).unwrap(),
      })
    }
  )
}

fn make_latent_decompressors(
  chunk_meta: &ChunkMeta,
  page_meta: &PageMeta,
  n: usize,
) -> PcoResult<PerLatentVar<DynLatentPageDecompressor>> {
  let mut states = PerLatentVarBuilder::default();
  for (key, (chunk_latent_var_meta, page_latent_var_meta)) in chunk_meta
    .per_latent_var
    .as_ref()
    .zip_exact(page_meta.per_latent_var.as_ref())
    .enumerated()
  {
    let var_delta_encoding = chunk_meta.delta_encoding.for_latent_var(key);
    let n_in_body = n.saturating_sub(var_delta_encoding.n_latents_per_state());
    let state = match_latent_enum!(
      &chunk_latent_var_meta.bins,
      DynBins<L>(bins) => {
        let delta_state = page_latent_var_meta
          .delta_state
          .downcast_ref::<L>()
          .unwrap()
          .clone();

        if bins.is_empty() && n_in_body > 0 {
          return Err(PcoError::corruption(format!(
            "unable to decompress chunk with no bins and {} latents",
            n_in_body
          )));
        }

        DynLatentPageDecompressor::create(
          chunk_latent_var_meta.ans_size_log,
          bins,
          var_delta_encoding,
          page_latent_var_meta.ans_final_state_idxs,
          delta_state,
        )?
      }
    );

    states.set(key, state);
  }
  Ok(states.into())
}

impl<R: BetterBufRead> PageDecompressorInner<R> {
  pub(crate) fn new(mut src: R, chunk_meta: &ChunkMeta, n: usize) -> PcoResult<Self> {
    bit_reader::ensure_buf_read_capacity(&mut src, PERFORMANT_BUF_READ_CAPACITY);
    let mut reader_builder = BitReaderBuilder::new(src, PAGE_PADDING, 0);

    let page_meta =
      reader_builder.with_reader(|reader| unsafe { PageMeta::read_from(reader, chunk_meta) })?;

    let mode = chunk_meta.mode;
    let latent_decompressors = make_latent_decompressors(chunk_meta, &page_meta, n)?;

    let delta_scratch = make_latent_scratch(latent_decompressors.delta.as_ref());
    let secondary_scratch = make_latent_scratch(latent_decompressors.secondary.as_ref());

    // we don't store the whole ChunkMeta because it can get large due to bins
    Ok(Self {
      n,
      mode,
      delta_encoding: chunk_meta.delta_encoding,
      reader_builder,
      n_processed: 0,
      latent_decompressors,
      delta_scratch,
      secondary_scratch,
    })
  }

  fn n_remaining(&self) -> usize {
    self.n - self.n_processed
  }
}

impl<T: Number, R: BetterBufRead> PageDecompressor<T, R> {
  #[inline(never)]
  pub(crate) fn new(src: R, chunk_meta: &ChunkMeta, n: usize) -> PcoResult<Self> {
    Ok(Self {
      inner: PageDecompressorInner::new(src, chunk_meta, n)?,
      phantom: PhantomData::<T>,
    })
  }

  fn decompress_batch(&mut self, dst: &mut [T]) -> PcoResult<()> {
    let batch_n = dst.len();
    let inner = &mut self.inner;
    let n = inner.n;
    let n_remaining = inner.n_remaining();
    let mode = inner.mode;

    // DELTA LATENTS
    if let Some(LatentScratch {
      is_constant: false,
      dst,
    }) = &mut inner.delta_scratch
    {
      let dyn_lpd = inner.latent_decompressors.delta.as_mut().unwrap();
      let limit = min(
        n_remaining.saturating_sub(inner.delta_encoding.n_latents_per_state()),
        batch_n,
      );
      inner.reader_builder.with_reader(|reader| unsafe {
        match_latent_enum!(
          dyn_lpd,
          DynLatentPageDecompressor<L>(lpd) => {
            // Delta latents only line up with pre-delta length of the other
            // latents.
            // We never apply delta encoding to delta latents, so we just
            // skip straight to the inner LatentBatchDecompressor
            lpd.decompress_batch_pre_delta(
              reader,
              &mut dst.downcast_mut::<L>().unwrap()[..limit]
            )
          }
        );
        Ok(())
      })?;
    }
    let delta_latents = inner.delta_scratch.as_ref().map(|scratch| &scratch.dst);

    // PRIMARY LATENTS
    inner.reader_builder.with_reader(|reader| unsafe {
      let primary_dst = T::transmute_to_latents(dst);
      let dyn_lpd = inner
        .latent_decompressors
        .primary
        .downcast_mut::<T::L>()
        .unwrap();
      dyn_lpd.decompress_batch(
        delta_latents,
        n_remaining,
        reader,
        primary_dst,
      )
    })?;

    // SECONDARY LATENTS
    if let Some(LatentScratch {
      is_constant: false,
      dst,
    }) = &mut inner.secondary_scratch
    {
      let dyn_lpd = inner.latent_decompressors.secondary.as_mut().unwrap();
      inner.reader_builder.with_reader(|reader| unsafe {
        match_latent_enum!(
          dyn_lpd,
          DynLatentPageDecompressor<L>(lpd) => {
            // We never apply delta encoding to delta latents, so we just
            // skip straight to the inner LatentBatchDecompressor
            lpd.decompress_batch(
              delta_latents,
              n_remaining,
              reader,
              &mut dst.downcast_mut::<L>().unwrap()[..batch_n]
            )
          }
        )
      })?;
    }

    T::join_latents(
      mode,
      T::transmute_to_latents(dst),
      inner.secondary_scratch.as_ref().map(|scratch| &scratch.dst),
    );
    convert_from_latents_to_numbers(dst);

    inner.n_processed += batch_n;
    if inner.n_processed == n {
      inner.reader_builder.with_reader(|reader| {
        reader.drain_empty_byte("expected trailing bits at end of page to be empty")
      })?;
    }

    Ok(())
  }

  /// Reads the next decompressed numbers into the destination, returning
  /// progress into the page and advancing along the compressed data.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// `dst` must have length either a multiple of 256 or be at least the count
  /// of numbers remaining in the page.
  pub fn decompress(&mut self, num_dst: &mut [T]) -> PcoResult<Progress> {
    let n_remaining = self.inner.n_remaining();
    if num_dst.len() % FULL_BATCH_N != 0 && num_dst.len() < n_remaining {
      return Err(PcoError::invalid_argument(format!(
        "num_dst's length must either be a multiple of {} or be \
         at least the count of numbers remaining ({} < {})",
        FULL_BATCH_N,
        num_dst.len(),
        n_remaining,
      )));
    }

    let n_to_process = min(num_dst.len(), n_remaining);

    let mut n_processed = 0;
    while n_processed < n_to_process {
      let dst_batch_end = min(n_processed + FULL_BATCH_N, n_to_process);
      self.decompress_batch(&mut num_dst[n_processed..dst_batch_end])?;
      n_processed = dst_batch_end;
    }

    Ok(Progress {
      n_processed,
      finished: self.inner.n_remaining() == 0,
    })
  }

  /// Returns the rest of the compressed data source.
  pub fn into_src(self) -> R {
    self.inner.reader_builder.into_inner()
  }
}
