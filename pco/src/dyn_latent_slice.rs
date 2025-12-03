use crate::data_types::Latent;

// Unfortunately we can't implement this with dtype_dispatch because it doesn't
// handle extra generics/lifetime parameters yet.
pub enum DynLatentSlice<'a> {
  U16(&'a [u16]),
  U32(&'a [u32]),
  U64(&'a [u64]),
}

impl<'a> DynLatentSlice<'a> {
  #[inline]
  pub fn downcast<L: Latent>(self) -> Option<&'a [L]> {
    match self {
      Self::U16(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u16>() => {
        let ptr = inner as *const [u16] as *const [L];
        Some(unsafe { &*ptr })
      }
      Self::U32(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u32>() => {
        let ptr = inner as *const [u32] as *const [L];
        Some(unsafe { &*ptr })
      }
      Self::U64(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u64>() => {
        let ptr = inner as *const [u64] as *const [L];
        Some(unsafe { &*ptr })
      }
      _ => None,
    }
  }
}
