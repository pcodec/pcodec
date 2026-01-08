use std::any;

use crate::data_types::Latent;

// Unfortunately we can't implement this with dtype_dispatch because it doesn't
// handle extra generics/lifetime parameters yet.
pub enum DynLatentSlice<'a> {
  U16(&'a mut [u16]),
  U32(&'a mut [u32]),
  U64(&'a mut [u64]),
}

impl<'a> DynLatentSlice<'a> {
  pub fn new<L: Latent>(inner: &mut [L]) -> Self {
    if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u16>() {
      let ptr = inner as *mut [L] as *mut [u16];
      Self::U16(unsafe { &mut *ptr })
    } else if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u32>() {
      let ptr = inner as *mut [L] as *mut [u32];
      Self::U32(unsafe { &mut *ptr })
    } else if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u64>() {
      let ptr = inner as *mut [L] as *mut [u64];
      Self::U64(unsafe { &mut *ptr })
    } else {
      unreachable!()
    }
  }

  pub fn downcast_unwrap<L: Latent>(self) -> &'a mut [L] {
    match self {
      Self::U16(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u16>() => {
        let ptr = inner as *mut [u16] as *mut [L];
        unsafe { &mut *ptr }
      }
      Self::U32(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u32>() => {
        let ptr = inner as *mut [u32] as *mut [L];
        unsafe { &mut *ptr }
      }
      Self::U64(inner) if std::any::TypeId::of::<L>() == std::any::TypeId::of::<u64>() => {
        let ptr = inner as *mut [u64] as *mut [L];
        unsafe { &mut *ptr }
      }
      _ => panic!(
        "attempted to downcast DynLatentSlice of wrong variant to {}; this is a bug in Pco",
        any::type_name::<L>(),
      ),
    }
  }
}
