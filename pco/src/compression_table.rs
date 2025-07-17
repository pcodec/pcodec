use crate::compression_intermediates::BinCompressionInfo;
use crate::data_types::Latent;

#[derive(Clone, Debug)]
pub struct CompressionTable<L: Latent> {
  pub search_size_log: usize,
  pub search_lowers: Vec<L>,
  pub infos: Vec<BinCompressionInfo<L>>,
}

impl<L: Latent> From<Vec<BinCompressionInfo<L>>> for CompressionTable<L> {
  fn from(mut infos: Vec<BinCompressionInfo<L>>) -> Self {
    infos.sort_unstable_by_key(|info| info.lower);

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
    self.infos.len() <= 1
  }
}
