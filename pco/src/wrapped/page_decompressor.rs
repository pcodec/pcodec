use std::cmp::min;
use std::fmt::Debug;

use better_io::BetterBufRead;

use crate::bit_reader;
use crate::bit_reader::BitReaderBuilder;
use crate::chunk_latent_decompressor::DynChunkLatentDecompressor;
use crate::constants::{FULL_BATCH_N, PAGE_PADDING};
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::page::PageMeta;
use crate::metadata::per_latent_var::{PerLatentVar, PerLatentVarBuilder};
use crate::metadata::DynLatents;
use crate::page_latent_decompressor::{DynPageLatentDecompressor, PageLatentDecompressor};
use crate::progress::Progress;
use crate::wrapped::chunk_decompressor::ChunkDecompressorInner;
use crate::wrapped::ChunkDecompressor;

const PERFORMANT_BUF_READ_CAPACITY: usize = 8192;

#[derive(Debug)]
struct LatentScratch {
  is_constant: bool,
  dst: DynLatents,
}

pub(crate) struct PageDecompressorState<R: BetterBufRead> {
  reader_builder: BitReaderBuilder<R>,
  n_remaining: usize,
  latent_decompressors: PerLatentVar<DynPageLatentDecompressor>,
  delta_scratch: Option<LatentScratch>,
  secondary_scratch: Option<LatentScratch>,
}

/// Holds metadata about a page and supports decompression.
pub struct PageDecompressor<'a, T: Number, R: BetterBufRead> {
  cd: &'a ChunkDecompressor<T>,
  state: PageDecompressorState<R>,
}

fn convert_from_latents_to_numbers<T: Number>(dst: &mut [T]) {
  // we wrote the joined latents to dst, so we can convert them in place
  for l_and_dst in dst {
    *l_and_dst = T::from_latent_ordered(l_and_dst.transmute_to_latent());
  }
}

fn make_latent_scratch(cld: Option<&DynChunkLatentDecompressor>) -> Option<LatentScratch> {
  let cld = cld?;

  match_latent_enum!(
    cld,
    DynChunkLatentDecompressor<L>(inner) => {
      let maybe_constant_value = inner.maybe_constant_value;
      Some(LatentScratch {
        is_constant: maybe_constant_value.is_some(),
        dst: DynLatents::new(vec![maybe_constant_value.unwrap_or_default(); FULL_BATCH_N]).unwrap(),
      })
    }
  )
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
        )?)).unwrap()
      }
    );

    builder.set(key, state);
  }
  Ok(builder.into())
}

impl<R: BetterBufRead> PageDecompressorState<R> {
  pub(crate) fn new(mut src: R, cd: &ChunkDecompressorInner, n: usize) -> PcoResult<Self> {
    bit_reader::ensure_buf_read_capacity(&mut src, PERFORMANT_BUF_READ_CAPACITY);
    let mut reader_builder = BitReaderBuilder::new(src, PAGE_PADDING, 0);

    let page_meta =
      reader_builder.with_reader(|reader| unsafe { PageMeta::read_from(reader, &cd.meta) })?;

    let latent_decompressors = make_latent_decompressors(cd, &page_meta, n)?;

    let delta_scratch = make_latent_scratch(cd.per_latent_var.delta.as_ref());
    let secondary_scratch = make_latent_scratch(cd.per_latent_var.secondary.as_ref());

    // we don't store the whole ChunkMeta because it can get large due to bins
    Ok(Self {
      reader_builder,
      n_remaining: n,
      latent_decompressors,
      delta_scratch,
      secondary_scratch,
    })
  }

  fn decompress_batch<T: Number>(
    &mut self,
    cd: &ChunkDecompressorInner,
    dst: &mut [T],
  ) -> PcoResult<()> {
    let batch_n = dst.len();
    let n_remaining = self.n_remaining;

    // DELTA LATENTS
    if let Some(LatentScratch {
      is_constant: false,
      dst,
    }) = &mut self.delta_scratch
    {
      let dyn_pld = self.latent_decompressors.delta.as_mut().unwrap();
      let limit = min(
        n_remaining.saturating_sub(cd.n_latents_per_delta_state()),
        batch_n,
      );
      self.reader_builder.with_reader(|reader| unsafe {
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
              &mut dst.downcast_mut::<L>().unwrap()[..limit]
            )
          }
        );
        Ok(())
      })?;
    }
    let delta_latents = self.delta_scratch.as_ref().map(|scratch| &scratch.dst);

    // PRIMARY LATENTS
    self.reader_builder.with_reader(|reader| unsafe {
      let primary_dst = T::transmute_to_latents(dst);
      let dyn_pld = self
        .latent_decompressors
        .primary
        .downcast_mut::<T::L>()
        .unwrap();
      dyn_pld.decompress_batch(
        reader,
        cd.per_latent_var.primary.downcast_ref::<T::L>().unwrap(),
        delta_latents,
        n_remaining,
        primary_dst,
      )
    })?;

    // SECONDARY LATENTS
    if let Some(LatentScratch {
      is_constant: false,
      dst,
    }) = &mut self.secondary_scratch
    {
      let dyn_pld = self.latent_decompressors.secondary.as_mut().unwrap();
      self.reader_builder.with_reader(|reader| unsafe {
        match_latent_enum!(
          dyn_pld,
          DynPageLatentDecompressor<L>(pld) => {
            // We never apply delta encoding to delta latents, so we just
            // skip straight to the self PageLatentDecompressor
            pld.decompress_batch(
              reader,
              cd.per_latent_var.secondary.as_ref().unwrap().downcast_ref::<L>().unwrap(),
              delta_latents,
              n_remaining,
              &mut dst.downcast_mut::<L>().unwrap()[..batch_n]
            )
          }
        )
      })?;
    }

    T::join_latents(
      &cd.meta.mode,
      T::transmute_to_latents(dst),
      self.secondary_scratch.as_ref().map(|scratch| &scratch.dst),
    );
    convert_from_latents_to_numbers(dst);

    self.n_remaining -= batch_n;
    if self.n_remaining == 0 {
      self.reader_builder.with_reader(|reader| {
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
