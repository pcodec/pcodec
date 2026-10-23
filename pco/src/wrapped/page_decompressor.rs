use std::cmp::min;
use std::marker::PhantomData;
use std::ops::Range;

use better_io::BetterBufRead;

use crate::bit_reader::{BitReader, BitReaderBuilder};
use crate::chunk_latent_decompressor::{
  DynChunkLatentDecompressor, DynChunkLatentDecompressorScratch,
};
use crate::constants::{FULL_BATCH_N, MAX_BATCH_LATENT_VAR_SIZE, OVERSHOOT_PADDING};
use crate::data_types::number_priv::NumberPriv;
use crate::data_types::{Latent, Number};
use crate::dyn_slices::{DynLatentSlice, DynNumberSliceMut};
use crate::errors::{PcoError, PcoResult};
use crate::macros::{match_latent_enum, match_number_enum};
use crate::metadata::page::PageMeta;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::ChunkMeta;
use crate::page_latent_decompressor::{DynPageLatentDecompressor, PageLatentDecompressor};
use crate::progress::Progress;
use crate::wrapped::chunk_decompressor::ChunkDecompressorInner;
use crate::wrapped::{ChunkDecompressor, DecompressorScratch};

pub(crate) struct PageDecompressorState<R: BetterBufRead> {
  reader_builder: BitReaderBuilder<R>,
  n_remaining: usize,
  latent_decompressors: PerLatentVar<DynPageLatentDecompressor>,
}

enum ChunkAndScratch<'a> {
  Own(&'a mut ChunkDecompressorInner),
  Caller(
    &'a ChunkDecompressorInner,
    &'a mut DecompressorScratch,
  ),
}

enum LatentVarAndScratch<'a> {
  Own(&'a mut DynChunkLatentDecompressor),
  Caller(
    &'a DynChunkLatentDecompressor,
    &'a mut DynChunkLatentDecompressorScratch,
  ),
}

impl ChunkAndScratch<'_> {
  fn inner(&self) -> &ChunkDecompressorInner {
    match self {
      Self::Own(cd) => cd,
      Self::Caller(cd, _) => cd,
    }
  }

  fn split(
    &mut self,
  ) -> (
    &ChunkMeta,
    PerLatentVar<LatentVarAndScratch<'_>>,
  ) {
    match self {
      Self::Own(cd) => (
        &cd.meta,
        cd.per_latent_var
          .as_mut()
          .map(|_key, cld| LatentVarAndScratch::Own(cld)),
      ),
      Self::Caller(cd, scratch) => (
        &cd.meta,
        cd.per_latent_var
          .as_ref()
          .zip_exact(scratch.per_latent_var.as_mut())
          .map(|_key, (cld, scratch)| LatentVarAndScratch::Caller(cld, scratch)),
      ),
    }
  }
}

impl LatentVarAndScratch<'_> {
  unsafe fn read_batch_pre_delta<L: Latent>(
    &mut self,
    pld: &mut PageLatentDecompressor<L>,
    reader: &mut BitReader,
    batch_n: usize,
  ) {
    match self {
      Self::Own(dyn_cld) => pld.read_batch_pre_delta(
        reader,
        batch_n,
        dyn_cld.downcast_mut::<L>().unwrap(),
      ),
      Self::Caller(dyn_cld, dyn_scratch) => pld.read_batch_pre_delta_into(
        reader,
        batch_n,
        &dyn_cld.downcast_ref::<L>().unwrap().view(),
        dyn_scratch.downcast_mut::<L>().unwrap(),
      ),
    }
  }

  unsafe fn read_batch<L: Latent>(
    &mut self,
    pld: &mut PageLatentDecompressor<L>,
    reader: &mut BitReader,
    delta_latents: Option<DynLatentSlice>,
    n_remaining: usize,
  ) -> PcoResult<()> {
    match self {
      Self::Own(dyn_cld) => pld.read_batch(
        reader,
        delta_latents,
        n_remaining,
        dyn_cld.downcast_mut::<L>().unwrap(),
      ),
      Self::Caller(dyn_cld, dyn_scratch) => pld.read_batch_into(
        reader,
        delta_latents,
        n_remaining,
        &dyn_cld.downcast_ref::<L>().unwrap().view(),
        dyn_scratch.downcast_mut::<L>().unwrap(),
      ),
    }
  }

  // past the n latents this batch wrote:
  //   Own    -> only this chunk's, so the full slice as before
  //   Caller -> maybe another chunk's, so only the first n
  fn latents(&mut self, n: usize) -> DynLatentSlice<'_> {
    match self {
      Self::Own(dyn_cld) => dyn_cld.latents(),
      Self::Caller(_, dyn_scratch) => dyn_scratch.latents(n),
    }
  }
}

/// Holds metadata about a page and supports decompression.
pub struct PageDecompressor<'a, T: Number, R: BetterBufRead> {
  cd: ChunkAndScratch<'a>,
  state: PageDecompressorState<R>,
  phantom: PhantomData<T>,
}

fn make_latent_decompressors(
  cd: &ChunkDecompressorInner,
  page_meta: &PageMeta,
  n: usize,
) -> PcoResult<PerLatentVar<DynPageLatentDecompressor>> {
  let n_in_body = n.saturating_sub(cd.n_latents_per_delta_state());
  cd.per_latent_var
    .as_ref()
    .zip_exact(page_meta.per_latent_var.as_ref())
    .map_result(|_key, (dyn_cld, page_latent_var_meta)| {
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

          let pld = PageLatentDecompressor::new(
            page_latent_var_meta.ans_final_state_idxs,
            &cld.delta_encoding,
            delta_state,
          );
          DynPageLatentDecompressor::new(pld)
        }
      );
      Ok(state)
    })
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

fn read_primary_or_secondary<'a, R: BetterBufRead>(
  reader_builder: &mut BitReaderBuilder<R>,
  delta_latents: Option<DynLatentSlice>,
  n_remaining: usize,
  latent_var: &'a mut LatentVarAndScratch,
  dyn_pld: &mut DynPageLatentDecompressor,
) -> PcoResult<DynLatentSlice<'a>> {
  reader_builder.with_reader(MAX_BATCH_LATENT_VAR_SIZE, |reader| unsafe {
    match_latent_enum!(
      dyn_pld,
      DynPageLatentDecompressor<L>(pld) => {
        latent_var.read_batch(pld, reader, delta_latents, n_remaining)
      }
    )
  })?;
  Ok(latent_var.latents(n_remaining.min(FULL_BATCH_N)))
}

impl<R: BetterBufRead> PageDecompressorState<R> {
  fn read_batch(
    &mut self,
    meta: &ChunkMeta,
    n_latents_per_delta_state: usize,
    latent_vars: &mut PerLatentVar<LatentVarAndScratch>,
    range: Range<usize>,
    dst: &mut DynNumberSliceMut,
  ) -> PcoResult<()> {
    let batch_n = range.len();
    let n_remaining = self.n_remaining;
    let limit = min(
      n_remaining.saturating_sub(n_latents_per_delta_state),
      batch_n,
    );
    let PerLatentVar {
      delta: delta_var,
      primary: primary_var,
      secondary: secondary_var,
    } = latent_vars;

    // DELTA LATENTS
    if let Some(dyn_pld) = self.latent_decompressors.delta.as_mut() {
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
              delta_var.as_mut().unwrap().read_batch_pre_delta(pld, reader, limit)
            }
          );
          Ok(())
        })?;
    }

    // PRIMARY AND SECONDARY LATENTS
    let primary = read_primary_or_secondary(
      &mut self.reader_builder,
      delta_var.as_mut().map(|delta| delta.latents(limit)),
      n_remaining,
      primary_var,
      &mut self.latent_decompressors.primary,
    )?;

    let secondary = match self.latent_decompressors.secondary.as_mut() {
      Some(dyn_pld) => Some(read_primary_or_secondary(
        &mut self.reader_builder,
        delta_var.as_mut().map(|delta| delta.latents(limit)),
        n_remaining,
        secondary_var.as_mut().unwrap(),
        dyn_pld,
      )?),
      None => None,
    };

    match_number_enum!(
      dst,
      DynNumberSliceMut<T>(dst) => {
        T::join_latents(
          &meta.mode,
          primary,
          secondary,
          &mut dst[range],
        )?;
      }
    );

    self.n_remaining -= batch_n;
    if self.n_remaining == 0 {
      self.reader_builder.with_reader(1, |reader| {
        reader.drain_empty_byte("expected trailing bits at end of page to be empty")
      })?;
    }

    Ok(())
  }

  pub fn read(
    &mut self,
    cd: &mut ChunkDecompressorInner,
    mut dst: DynNumberSliceMut,
  ) -> PcoResult<Progress> {
    self.read_with_scratch(&mut ChunkAndScratch::Own(cd), &mut dst)
  }

  fn read_with_scratch(
    &mut self,
    cd: &mut ChunkAndScratch,
    dst: &mut DynNumberSliceMut,
  ) -> PcoResult<Progress> {
    let n_remaining = self.n_remaining;
    let dst_len = dst.len();
    if !dst_len.is_multiple_of(FULL_BATCH_N) && dst_len < n_remaining {
      return Err(PcoError::invalid_argument(format!(
        "num_dst's length must either be a multiple of {} or be \
         at least the count of numbers remaining ({} < {})",
        FULL_BATCH_N, dst_len, n_remaining,
      )));
    }

    let n_to_process = min(dst_len, self.n_remaining);
    let n_latents_per_delta_state = cd.inner().n_latents_per_delta_state();
    let (meta, mut latent_vars) = cd.split();

    let mut n_processed = 0;
    while n_processed < n_to_process {
      let dst_batch_end = min(n_processed + FULL_BATCH_N, n_to_process);
      self.read_batch(
        meta,
        n_latents_per_delta_state,
        &mut latent_vars,
        n_processed..dst_batch_end,
        dst,
      )?;
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
  pub(crate) fn new(src: R, cd: &'a mut ChunkDecompressor<T>, n: usize) -> PcoResult<Self> {
    let state = PageDecompressorState::new(src, &cd.inner, n)?;
    Ok(Self {
      cd: ChunkAndScratch::Own(&mut cd.inner),
      state,
      phantom: PhantomData,
    })
  }

  #[inline(never)]
  pub(crate) fn new_with_scratch(
    src: R,
    cd: &'a ChunkDecompressor<T>,
    scratch: &'a mut DecompressorScratch,
    n: usize,
  ) -> PcoResult<Self> {
    let state = PageDecompressorState::new(src, &cd.inner, n)?;
    // the scratch may come from another chunk
    cd.inner
      .per_latent_var
      .as_ref()
      .zip_exact(scratch.per_latent_var.as_mut())
      .map(|_key, (cld, scratch)| cld.prepopulate(scratch));
    Ok(Self {
      cd: ChunkAndScratch::Caller(&cd.inner, scratch),
      state,
      phantom: PhantomData,
    })
  }

  /// Reads the next decompressed numbers into the destination, returning
  /// progress into the page and advancing along the compressed data.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// `dst` must have length either a multiple of 256 or be at least the count
  /// of numbers remaining in the page.
  pub fn read(&mut self, dst: &mut [T]) -> PcoResult<Progress> {
    self.state.read_with_scratch(
      &mut self.cd,
      &mut DynNumberSliceMut::new(dst),
    )
  }

  /// Returns the rest of the compressed data source.
  pub fn into_src(self) -> R {
    self.state.into_src()
  }
}
