use crate::ans::{self, Spec};
use crate::constants::Bitlen;
use crate::data_types::Latent;
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::{bins, Bin, ChunkLatentVarMeta, DynBins};
use crate::scratch_array::ScratchArray;
use crate::{read_write_uint, FULL_BATCH_N};

/// Per-batch working memory for decompressing one latent variable. It holds
/// no information about any chunk beyond the single-bin offsets it may be
/// prefilled with, so one scratch can serve pages of any chunk with the same
/// latent type.
#[derive(Clone, Debug)]
pub struct LatentScratch<L: Latent> {
  pub offset_bits_csum: ScratchArray<Bitlen>,
  pub offset_bits: ScratchArray<Bitlen>,
  pub latents: ScratchArray<L>,
  // The offset bit width that offset_bits and offset_bits_csum are currently
  // filled with for a single-bin chunk, or None after ANS decoding wrote
  // per-symbol widths into them.
  pub prefilled_offset_bits: Option<Bitlen>,
}

impl<L: Latent> LatentScratch<L> {
  pub fn new() -> Box<Self> {
    Box::new(Self {
      offset_bits_csum: ScratchArray([0; FULL_BATCH_N]),
      offset_bits: ScratchArray([0; FULL_BATCH_N]),
      latents: ScratchArray([L::ZERO; FULL_BATCH_N]),
      prefilled_offset_bits: None,
    })
  }

  // A single-bin chunk has the same offset width for every latent, so we set
  // the offset state once and keep it for as long as the same width is used.
  #[inline]
  pub fn prefill_single_bin(&mut self, offset_bits: Bitlen) {
    if self.prefilled_offset_bits == Some(offset_bits) {
      return;
    }

    let mut csum = 0;
    for i in 0..FULL_BATCH_N {
      self.offset_bits[i] = offset_bits;
      self.offset_bits_csum[i] = csum;
      csum += offset_bits;
    }
    self.prefilled_offset_bits = Some(offset_bits);
  }
}

// we allocate these on the heap because they're enormous
type BoxedScratch<L> = Box<LatentScratch<L>>;

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynLatentScratch(BoxedScratch)
);

impl DynLatentScratch {
  pub fn latents(&self) -> DynLatentSlice<'_> {
    match_latent_enum!(
      self,
      DynLatentScratch<L>(inner) => {
        DynLatentSlice::new(&*inner.latents)
      }
    )
  }
}

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressor<L: Latent> {
  pub delta_encoding: LatentVarDeltaEncoding,
  pub bytes_per_offset: usize,
  pub state_lowers: Vec<L>,
  pub n_bins: usize,
  pub decoder: ans::Decoder,
  // offset width of the only bin, used when n_bins <= 1
  pub only_bin_offset_bits: Bitlen,
}

impl<L: Latent> ChunkLatentDecompressor<L> {
  pub fn new(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: LatentVarDeltaEncoding,
  ) -> PcoResult<Box<Self>> {
    let bytes_per_offset = read_write_uint::calc_max_bytes(bins::max_offset_bits(bins));
    let bin_offset_bits = bins.iter().map(|bin| bin.offset_bits).collect::<Vec<_>>();
    let weights = bins::weights(bins);
    let ans_spec = Spec::from_weights(ans_size_log, weights)?;
    let state_lowers = ans_spec
      .state_symbols
      .iter()
      .map(|&s| bins.get(s as usize).map_or(L::ZERO, |b| b.lower))
      .collect();
    let decoder = ans::Decoder::new(&ans_spec, &bin_offset_bits);

    let only_bin_offset_bits = if bins.len() == 1 {
      bins[0].offset_bits
    } else {
      0
    };

    Ok(Box::new(Self {
      bytes_per_offset,
      state_lowers,
      n_bins: bins.len(),
      decoder,
      delta_encoding,
      only_bin_offset_bits,
    }))
  }
}

// we allocate these on the heap because they're enormous
type Boxed<L> = Box<ChunkLatentDecompressor<L>>;

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynChunkLatentDecompressor(Boxed)
);

impl DynChunkLatentDecompressor {
  pub fn create(
    latent_var: &ChunkLatentVarMeta,
    delta_encoding: LatentVarDeltaEncoding,
  ) -> PcoResult<DynChunkLatentDecompressor> {
    let res = match_latent_enum!(
      &latent_var.bins,
      DynBins<L>(bins) => {
        let inner = ChunkLatentDecompressor::new(
          latent_var.ans_size_log,
          bins,
          delta_encoding,
        )?;
        DynChunkLatentDecompressor::new(inner)
      }
    );
    Ok(res)
  }

  pub fn new_scratch(&self) -> DynLatentScratch {
    match_latent_enum!(
      self,
      DynChunkLatentDecompressor<L>(_inner) => {
        DynLatentScratch::new(LatentScratch::<L>::new())
      }
    )
  }

  pub fn scratch_matches(&self, scratch: &DynLatentScratch) -> bool {
    match_latent_enum!(
      self,
      DynChunkLatentDecompressor<L>(_inner) => {
        scratch.downcast_ref::<L>().is_some()
      }
    )
  }
}
