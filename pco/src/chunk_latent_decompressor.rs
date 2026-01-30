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

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressorScratch<L: Latent> {
  pub offset_bits_csum: ScratchArray<Bitlen>,
  pub offset_bits: ScratchArray<Bitlen>,
  pub latents: ScratchArray<L>,
}

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressor<L: Latent> {
  pub delta_encoding: LatentVarDeltaEncoding,
  pub bytes_per_offset: usize,
  pub state_lowers: Vec<L>,
  pub n_bins: usize,
  pub decoder: ans::Decoder,
  pub scratch: ChunkLatentDecompressorScratch<L>,
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

    let only_bin = if bins.len() == 1 { Some(bins[0]) } else { None };

    let mut offset_bits_csum = ScratchArray([0; FULL_BATCH_N]);
    let mut offset_bits = ScratchArray([0; FULL_BATCH_N]);
    let mut latents = ScratchArray([L::ZERO; FULL_BATCH_N]);

    if let Some(bin) = &only_bin {
      // we optimize performance by setting state once and never again
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        offset_bits[i] = bin.offset_bits;
        offset_bits_csum[i] = csum;
        latents[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    Ok(Box::new(Self {
      bytes_per_offset,
      state_lowers,
      n_bins: bins.len(),
      decoder,
      delta_encoding,
      scratch: ChunkLatentDecompressorScratch {
        offset_bits_csum,
        offset_bits,
        latents,
      },
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

  pub fn latents<'a>(&'a mut self) -> DynLatentSlice<'a> {
    match_latent_enum!(
      self,
      DynChunkLatentDecompressor<L>(inner) => {
        DynLatentSlice::new(&*inner.scratch.latents)
      }
    )
  }
}
