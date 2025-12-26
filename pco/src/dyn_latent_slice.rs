use crate::data_types::Latent;

// Unfortunately we can't implement this with dtype_dispatch because it doesn't
// handle extra generics/lifetime parameters yet.
pub enum DynLatentSlice<'a> {
  U16(&'a mut [u16]),
  U32(&'a mut [u32]),
  U64(&'a mut [u64]),
}

impl<'a> DynLatentSlice<'a> {
  pub fn downcast<L: Latent>(self) -> Option<&'a mut [L]> {
    match self {
      Self::U16(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u16>() => {
        let ptr = inner as *mut [u16] as *mut [L];
        Some(unsafe { &mut *ptr })
      }
      Self::U32(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u32>() => {
        let ptr = inner as *mut [u32] as *mut [L];
        Some(unsafe { &mut *ptr })
      }
      Self::U64(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u64>() => {
        let ptr = inner as *mut [u64] as *mut [L];
        Some(unsafe { &mut *ptr })
      }
      _ => None,
    }
  }
}
