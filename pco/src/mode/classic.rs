use crate::data_types::{Number, SplitLatents};
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::DynLatents;

pub(crate) fn split_latents<T: Number>(nums: &[T]) -> SplitLatents {
  let primary = DynLatents::new(nums.iter().map(|&x| x.to_latent_ordered()).collect());
  SplitLatents {
    primary,
    secondary: None,
  }
}

pub(crate) fn join_latents<T: Number>(primary: DynLatentSlice, dst: &mut [T]) -> PcoResult<()> {
  for (&l, num) in primary
    .downcast::<T::L>()
    .unwrap()
    .iter()
    .zip(dst.iter_mut())
  {
    *num = T::from_latent_ordered(l);
  }
  Ok(())
}

#[cfg(feature = "bench")]
mod micro {
  use divan::{black_box, Bencher};

  use super::*;
  use crate::bench_utils::{random_walk_nums, BENCH_N};
  use crate::constants::FULL_BATCH_N;
  use crate::data_types::Latent;

  #[divan::bench(types = [u8, u16, u32, u64])]
  fn join_latents<T: Number<L = T> + Latent>(bencher: Bencher) {
    let nums = random_walk_nums::<T>();
    let primary = split_latents(&nums).primary.downcast::<T>().unwrap();
    let mut dst = vec![T::ZERO; nums.len()];
    bencher
      .counter(divan::counter::ItemsCount::new(BENCH_N))
      .bench_local(|| {
        for (batch_idx, dst_batch) in dst.chunks_mut(FULL_BATCH_N).enumerate() {
          let start = batch_idx * FULL_BATCH_N;
          super::join_latents(
            black_box(DynLatentSlice::new(
              &primary[start..start + dst_batch.len()],
            )),
            dst_batch,
          )
          .unwrap();
        }
      });
  }
}
