use crate::{
  data_types::{Number, SplitLatents},
  dyn_latent_slice::DynLatentSlice,
  metadata::DynLatents,
};

pub(crate) fn split_latents<T: Number>(nums: &[T]) -> SplitLatents {
  let primary = DynLatents::new(nums.iter().map(|&x| x.to_latent_ordered()).collect());
  SplitLatents {
    primary,
    secondary: None,
  }
}

pub(crate) fn join_latents<T: Number>(primary: DynLatentSlice, dst: &mut [T]) {
  for (&l, num) in primary.downcast_unwrap::<T::L>().iter().zip(dst.iter_mut()) {
    *num = T::from_latent_ordered(l);
  }
}
