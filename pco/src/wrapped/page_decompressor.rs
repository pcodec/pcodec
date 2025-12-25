use std::cmp::min;

use better_io::BetterBufRead;

use crate::bit_reader::BitReaderBuilder;
use crate::chunk_latent_decompressor::DynChunkLatentDecompressor;
use crate::constants::{FULL_BATCH_N, MAX_BATCH_LATENT_VAR_SIZE, OVERSHOOT_PADDING};
use crate::data_types::Number;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::page::PageMeta;
use crate::metadata::per_latent_var::{PerLatentVar, PerLatentVarBuilder};
use crate::page_latent_decompressor::DynPageLatentDecompressor;
use crate::page_latent_decompressor::PageLatentDecompressor;
use crate::progress::Progress;
use crate::wrapped::chunk_decompressor::ChunkDecompressorInner;
use crate::wrapped::ChunkDecompressor;

pub(crate) struct PageDecompressorState<R: BetterBufRead> {
  reader_builder: BitReaderBuilder<R>,
  n_remaining: usize,
  latent_decompressors: PerLatentVar<DynPageLatentDecompressor>,
}

/// Holds metadata about a page and supports decompression.
pub struct PageDecompressor<'a, T: Number, R: BetterBufRead> {
  cd: &'a ChunkDecompressor<T>,
  state: PageDecompressorState<R>,
}

fn make_latent_decompressors(
  cd: &ChunkDecompressorInner,
  page_meta: &PageMeta,
  n: usize,
) -> PcoResult<PerLatentVar<DynPageLatentDecompressor>> {
  let mut builder = PerLatentVarBuilder::default();
  for (key, (dyn_cld, page_latent_var_meta)) in cd
    .per_latent_var
    .as_ref()
    .zip_exact(page_meta.per_latent_var.as_ref())
    .enumerated()
  {
    let n_in_body = n.saturating_sub(cd.n_latents_per_delta_state());
    let state = match_latent_enum!(
      &dyn_cld,
      DynChunkLatentDecompressor<L>(cld) => {
        let delta_state = page_latent_var_meta
          .delta_state
          .downcast_ref::<L>()
          .unwrap()
          .clone();

        if cld.n_bins == 0 && n_in_body > 0 {
          return Err(PcoError::corruption(format!(
            "unable to decompress chunk with no bins and {} latents",
            n_in_body
          )));
        }

        DynPageLatentDecompressor::new(Box::new(PageLatentDecompressor::new(
          cld,
          page_latent_var_meta.ans_final_state_idxs,
          delta_state,
        )?))
      }
    );

    builder.set(key, state);
  }
  Ok(builder.into())
}

impl<R: BetterBufRead> PageDecompressorState<R> {
  pub(crate) fn new(src: R, cd: &ChunkDecompressorInner, n: usize) -> PcoResult<Self> {
    let mut reader_builder = BitReaderBuilder::new(src);
    let page_meta = reader_builder.with_reader(
      cd.meta.exact_page_meta_size() + OVERSHOOT_PADDING,
      |reader| unsafe { PageMeta::read_from(reader, &cd.meta) },
    )?;

    let latent_decompressors = make_latent_decompressors(cd, &page_meta, n)?;

    // we don't store the whole ChunkMeta because it can get large due to bins
    Ok(Self {
      reader_builder,
      n_remaining: n,
      latent_decompressors,
    })
  }
}

fn decompress_primary_or_secondary<'a, R: BetterBufRead>(
  reader_builder: &mut BitReaderBuilder<R>,
  dyn_cld: &DynChunkLatentDecompressor,
  dyn_pld: &'a mut DynPageLatentDecompressor,
  delta_latents: Option<DynLatentSlice>,
  n_remaining: usize,
) -> PcoResult<DynLatentSlice<'a>> {
  reader_builder.with_reader(MAX_BATCH_LATENT_VAR_SIZE, |reader| unsafe {
    match_latent_enum!(
      dyn_pld,
      DynPageLatentDecompressor<L>(pld) => {
        let cld = dyn_cld.downcast_ref::<L>().unwrap();
        pld.decompress_batch(
          reader,
          cld,
          delta_latents,
          n_remaining,
        )
      }
    )
  })?;
  Ok(dyn_pld.latents())
}

impl<R: BetterBufRead> PageDecompressorState<R> {
  fn decompress_batch<T: Number>(
    &mut self,
    cd: &ChunkDecompressorInner,
    dst: &mut [T],
  ) -> PcoResult<()> {
    let batch_n = dst.len();
    let n_remaining = self.n_remaining;

    // DELTA LATENTS
    if let Some(dyn_pld) = self.latent_decompressors.delta.as_mut() {
      let limit = min(
        n_remaining.saturating_sub(cd.n_latents_per_delta_state()),
        batch_n,
      );
      self
        .reader_builder
        .with_reader(MAX_BATCH_LATENT_VAR_SIZE, |reader| unsafe {
          match_latent_enum!(
            dyn_pld,
            DynPageLatentDecompressor<L>(pld) => {
              // Delta latents only line up with pre-delta length of the other
              // latents.
              // We never apply delta encoding to delta latents, so we just
              // skip straight to the pre-delta routine.
              pld.decompress_batch_pre_delta(
                reader,
                cd.per_latent_var.delta.as_ref().unwrap().downcast_ref::<L>().unwrap(),
                limit,
              )
            }
          );
          Ok(())
        })?;
    }
    let delta_pld = &mut self.latent_decompressors.delta;

    // PRIMARY AND SECONDARY LATENTS
    let primary = decompress_primary_or_secondary(
      &mut self.reader_builder,
      &cd.per_latent_var.primary,
      &mut self.latent_decompressors.primary,
      delta_pld.as_mut().map(|pld| pld.latents()),
      n_remaining,
    )?;

    let secondary = match self.latent_decompressors.secondary.as_mut() {
      Some(dyn_pld) => Some(decompress_primary_or_secondary(
        &mut self.reader_builder,
        cd.per_latent_var.secondary.as_ref().unwrap(),
        dyn_pld,
        delta_pld.as_mut().map(|pld| pld.latents()),
        n_remaining,
      )?),
      None => None,
    };

    T::join_latents(&cd.meta.mode, primary, secondary, dst);

    self.n_remaining -= batch_n;
    if self.n_remaining == 0 {
      self.reader_builder.with_reader(1, |reader| {
        reader.drain_empty_byte("expected trailing bits at end of page to be empty")
      })?;
    }

    Ok(())
  }

  pub fn decompress<T: Number>(
    &mut self,
    cd: &ChunkDecompressorInner,
    num_dst: &mut [T],
  ) -> PcoResult<Progress> {
    let n_remaining = self.n_remaining;
    if !num_dst.len().is_multiple_of(FULL_BATCH_N) && num_dst.len() < n_remaining {
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
      self.decompress_batch(cd, &mut num_dst[n_processed..dst_batch_end])?;
      n_processed = dst_batch_end;
    }

    Ok(Progress {
      n_processed,
      finished: self.n_remaining == 0,
    })
  }

  pub fn into_src(self) -> R {
    self.reader_builder.into_inner()
  }
}

impl<'a, T: Number, R: BetterBufRead> PageDecompressor<'a, T, R> {
  #[inline(never)]
  pub(crate) fn new(src: R, cd: &'a ChunkDecompressor<T>, n: usize) -> PcoResult<Self> {
    Ok(Self {
      cd,
      state: PageDecompressorState::new(src, &cd.inner, n)?,
    })
  }

  /// Reads the next decompressed numbers into the destination, returning
  /// progress into the page and advancing along the compressed data.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// `dst` must have length either a multiple of 256 or be at least the count
  /// of numbers remaining in the page.
  pub fn decompress(&mut self, dst: &mut [T]) -> PcoResult<Progress> {
    self.state.decompress(&self.cd.inner, dst)
  }

  /// Returns the rest of the compressed data source.
  pub fn into_src(self) -> R {
    self.state.into_src()
  }
}
