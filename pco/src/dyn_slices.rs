use std::mem::MaybeUninit;
use std::ops::Range;
use std::slice;

use crate::data_types::{Latent, Number};
use crate::macros::{define_latent_enum, define_number_enum, match_number_enum};

type Arr<T> = [T];
type UninitArr<T> = [MaybeUninit<T>];

define_latent_enum!(
  #[derive()]
  pub DynLatentSlice(&Arr)
);

define_number_enum!(
  #[derive()]
  pub DynNumberSlice(&Arr)
);

impl<'a> DynNumberSlice<'a> {
  pub fn len(&self) -> usize {
    match_number_enum!(
      self,
      DynNumberSlice<T>(inner) => {
        inner.len()
      }
    )
  }

  pub fn slice(&self, range: Range<usize>) -> Self {
    match_number_enum!(
      self,
      DynNumberSlice<T>(inner) => {
        DynNumberSlice::new(&inner[range])
      }
    )
  }
}

define_number_enum!(
  #[derive()]
  pub DynNumberSliceMut(&mut UninitArr)
);

impl<'a> DynNumberSliceMut<'a> {
  pub fn len(&self) -> usize {
    match_number_enum!(
      self,
      DynNumberSliceMut<T>(inner) => {
        inner.len()
      }
    )
  }
}

/// View initialized numbers as possibly uninitialized slots.
///
/// # Safety
/// Every slot must remain initialized when the returned borrow ends, including
/// on error or unwind. Callers must not use the view to deinitialize a value.
#[inline]
pub(crate) unsafe fn as_uninit_mut<T: Number>(values: &mut [T]) -> &mut [MaybeUninit<T>] {
  // SAFETY: MaybeUninit<T> has the same size and alignment as T and permits
  // initialized values. The exclusive borrow and length are unchanged.
  unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}
