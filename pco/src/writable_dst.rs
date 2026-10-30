use std::mem::{self, MaybeUninit};

use crate::data_types::Number;

pub(crate) mod private {
  use std::mem::MaybeUninit;

  // The conversion lives on this private trait so that users cannot call it.
  // Otherwise safe code could write uninitialized values into a `&mut [T]`.
  pub trait WritableDstPriv<T> {
    fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
  }
}

/// A `[T]`-resembling that can receive decompressed numbers.
pub trait WritableDst<T: Number>: private::WritableDstPriv<T> {}

impl<T: Number, D: private::WritableDstPriv<T> + ?Sized> WritableDst<T> for D {}

impl<T: Number> private::WritableDstPriv<T> for [T] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    // Safety: MaybeUninit<T> has the same layout as T, and we only ever write
    // initialized values through the result.
    unsafe { mem::transmute(self) }
  }
}

impl<T: Number, const N: usize> private::WritableDstPriv<T> for [T; N] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    // Safety: MaybeUninit<T> has the same layout as T, and we only ever write
    // initialized values through the result.
    unsafe { mem::transmute(self.as_mut_slice()) }
  }
}

impl<T: Number> private::WritableDstPriv<T> for [MaybeUninit<T>] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    self
  }
}

impl<T: Number> private::WritableDstPriv<T> for Vec<T> {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    private::WritableDstPriv::as_maybe_uninit_slice_mut(self.as_mut_slice())
  }
}
