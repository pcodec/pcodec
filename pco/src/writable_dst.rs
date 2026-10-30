use std::mem::{self, MaybeUninit};

use crate::data_types::Number;

pub(crate) mod private {
  use std::mem::MaybeUninit;

  pub trait WritableDstPriv<T> {
    unsafe fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
  }
}

/// A `[T]`-resembling that can receive decompressed numbers.
pub trait WritableDst<T: Number>: private::WritableDstPriv<T> {}

impl<T: Number, D: private::WritableDstPriv<T> + ?Sized> WritableDst<T> for D {}

impl<T: Number> private::WritableDstPriv<T> for [T] {
  unsafe fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    mem::transmute(self)
  }
}

impl<T: Number, const N: usize> private::WritableDstPriv<T> for [T; N] {
  unsafe fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    mem::transmute(self.as_mut_slice())
  }
}

impl<T: Number> private::WritableDstPriv<T> for [MaybeUninit<T>] {
  unsafe fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    self
  }
}

impl<T: Number> private::WritableDstPriv<T> for Vec<T> {
  unsafe fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    mem::transmute(self.as_mut_slice())
  }
}
