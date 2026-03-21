use std::mem;

use crate::dtypes::PcoNumber;

// cursed ways to convert nums to bytes and back without doing work
pub unsafe fn num_slice_to_bytes<T: PcoNumber>(slice: &[T]) -> &[u8] {
  let byte_len = mem::size_of_val(slice);
  &*std::ptr::slice_from_raw_parts(
    mem::transmute::<*const T, *const u8>(slice.as_ptr()),
    byte_len,
  )
}
