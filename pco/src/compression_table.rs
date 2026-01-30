use std::cmp;

use crate::compression_intermediates::BinCompressionInfo;
use crate::data_types::Latent;
use crate::FULL_BATCH_N;

#[derive(Clone, Debug)]
pub struct CompressionTable<L: Latent> {
  search_size_log: usize,
  search_lowers: Vec<L>,
  pub infos: Vec<BinCompressionInfo<L>>,
}

impl<L: Latent> From<Vec<BinCompressionInfo<L>>> for CompressionTable<L> {
  fn from(mut infos: Vec<BinCompressionInfo<L>>) -> Self {
    let search_size_log = if infos.len() <= 1 {
      0
    } else {
      1 + (infos.len() - 1).ilog2() as usize
    };
    infos.sort_unstable_by_key(|info| info.lower);
    let mut search_lowers = infos.iter().map(|info| info.lower).collect::<Vec<_>>();
    while search_lowers.len() < (1 << search_size_log) {
      search_lowers.push(L::MAX);
    }

    Self {
      search_size_log,
      search_lowers,
      infos,
    }
  }
}

impl<L: Latent> CompressionTable<L> {
  pub fn is_trivial(&self) -> bool {
    // It's possible for the table to be trivial even when only_bin() is None;
    // the table can be empty
    self.infos.len() <= 1
  }

  pub fn only_bin(&self) -> Option<&BinCompressionInfo<L>> {
    if self.is_trivial() {
      self.infos.first()
    } else {
      None
    }
  }

  #[inline(never)]
  pub fn binary_search(&self, latents: &[L]) -> [usize; FULL_BATCH_N] {
    let mut search_idxs = [0; FULL_BATCH_N];

    // we do this as `size_log` SIMD loops over the batch
    for depth in 0..self.search_size_log {
      let bisection_idx = 1 << (self.search_size_log - 1 - depth);
      for (&latent, search_idx) in latents.iter().zip(search_idxs.iter_mut()) {
        let candidate_idx = *search_idx + bisection_idx;
        let value = unsafe { *self.search_lowers.get_unchecked(candidate_idx) };
        *search_idx += ((latent >= value) as usize) * bisection_idx;
      }
    }

    let n_bins = self.infos.len();
    if n_bins < 1 << self.search_size_log {
      // We worked with a balanced binary tree with missing leaves filled, so it
      // might have overshot some bin indices.
      search_idxs
        .iter_mut()
        .for_each(|search_idx| *search_idx = cmp::min(*search_idx, n_bins - 1));
    }

    search_idxs
  }
}
