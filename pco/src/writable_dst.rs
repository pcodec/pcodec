use std::mem::{self, MaybeUninit};

use crate::data_types::Number;

mod private {
  pub trait Sealed {}
}

/// A `[T]`-resembling that can receive decompressed numbers.
pub trait WritableDst<T: Number>: private::Sealed {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
}

impl<T: Number> private::Sealed for [T] {}
impl<T: Number> WritableDst<T> for [T] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    unsafe { mem::transmute(self) }
  }
}

impl<T: Number> private::Sealed for [MaybeUninit<T>] {}
impl<T: Number> WritableDst<T> for [MaybeUninit<T>] {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    self
  }
}

impl<T: Number> private::Sealed for Vec<T> {}
impl<T: Number> WritableDst<T> for Vec<T> {
  fn as_maybe_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
    self.as_mut_slice().as_maybe_uninit_slice_mut()
  }
}
