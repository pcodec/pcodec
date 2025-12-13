use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::{ChunkLatentVarMeta, ChunkMeta, DynBins, LatentVarKey};
use crate::{
  ans::{self, AnsState, Spec},
  constants::{Bitlen, ANS_INTERLEAVING},
  data_types::Latent,
  delta,
  errors::PcoResult,
  metadata::{bins, delta_encoding::LatentVarDeltaEncoding, Bin},
  read_write_uint, FULL_BATCH_N,
};

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressor<L: Latent> {
  pub delta_encoding: LatentVarDeltaEncoding,
  pub bytes_per_offset: usize,
  pub state_lowers: Vec<L>,
  pub n_bins: usize,
  pub only_bin: Option<Bin<L>>,
  pub decoder: ans::Decoder,
  pub maybe_constant_value: Option<L>,
}

impl<L: Latent> ChunkLatentDecompressor<L> {
  pub fn new(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: LatentVarDeltaEncoding,
  ) -> PcoResult<Self> {
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

    let maybe_constant_value =
      if bins::are_trivial(bins) && matches!(delta_encoding, LatentVarDeltaEncoding::NoOp) {
        bins.first().map(|bin| bin.lower)
      } else {
        None
      };

    let only_bin = if bins.len() == 1 { Some(bins[0]) } else { None };

    Ok(Self {
      bytes_per_offset,
      state_lowers,
      n_bins: bins.len(),
      only_bin,
      decoder,
      delta_encoding,
      maybe_constant_value,
    })
  }
}

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynChunkLatentDecompressor(ChunkLatentDecompressor)
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
          &bins,
          delta_encoding,
        )?;
        DynChunkLatentDecompressor::new(inner).unwrap()
      }
    );
    Ok(res)
  }
}
