use std::mem::{self, MaybeUninit};

use crate::data_types::Number;

mod private {
  use std::mem::MaybeUninit;

  // The conversion lives on this private trait so that users cannot call it.
  // Otherwise safe code could write uninitialized values into a `&mut [T]`.
  pub trait Sealed<T> {
    fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
  }
}

/// A `[T]`-resembling that can receive decompressed numbers.
pub trait WritableDst<T: Number>: private::Sealed<T> {}

pub(crate) fn as_maybe_uninit_slice_mut<T: Number, D: WritableDst<T> + ?Sized>(
  dst: &mut D,
) -> &mut [MaybeUninit<T>] {
  private::Sealed::as_maybe_uninit_slice_mut(dst)
}

impl<T: Number> private::Sealed<T> for [T] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    // Safety: MaybeUninit<T> has the same layout as T, and we only ever write
    // initialized values through the result.
    unsafe { mem::transmute(self) }
  }
}
impl<T: Number> WritableDst<T> for [T] {}

impl<T: Number> private::Sealed<T> for [MaybeUninit<T>] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    self
  }
}
impl<T: Number> WritableDst<T> for [MaybeUninit<T>] {}

impl<T: Number> private::Sealed<T> for Vec<T> {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    private::Sealed::as_maybe_uninit_slice_mut(self.as_mut_slice())
  }
}
impl<T: Number> WritableDst<T> for Vec<T> {}
