use crate::ans::{self, Spec};
use crate::constants::Bitlen;
use crate::data_types::Latent;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::{bins, Bin, ChunkLatentVarMeta, DynBins};
use crate::scratch::{DynBin, LatentDecompressorScratch};
use crate::{read_write_uint, FULL_BATCH_N};

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressor<L: Latent> {
  pub delta_encoding: LatentVarDeltaEncoding,
  pub bytes_per_offset: usize,
  pub state_lowers: Vec<L>,
  pub n_bins: usize,
  pub decoder: ans::Decoder,
  only_bin: Option<DynBin>,
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
    let only_bin = if bins.len() == 1 {
      Some(DynBin::new(bins[0]))
    } else {
      None
    };

    Ok(Self {
      bytes_per_offset,
      state_lowers,
      n_bins: bins.len(),
      decoder,
      delta_encoding,
      only_bin,
    })
  }

  pub fn adapt_scratch(&self, scratch: &mut LatentDecompressorScratch) {
    if self.only_bin.is_some() && scratch.only_bin != self.only_bin {
      // With a single bin, we optimize performance by setting offset bits once
      // so that we can skip ANS decoding in each batch.
      let bin = self.only_bin.unwrap().downcast::<L>().unwrap();
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        scratch.offset_bits[i] = bin.offset_bits;
        scratch.offset_bits_csum[i] = csum;
        csum += bin.offset_bits;
      }
    }

    scratch.only_bin = self.only_bin;
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
          bins,
          delta_encoding,
        )?;
        DynChunkLatentDecompressor::new(inner)
      }
    );
    Ok(res)
  }

  pub fn adapt_scratch(&self, scratch: &mut LatentDecompressorScratch) {
    match_latent_enum!(
      self,
      DynChunkLatentDecompressor<L>(cld) => {
        cld.adapt_scratch(scratch);
      }
    )
  }
}
