/// Information about progress after calling a decompression function.
#[derive(Clone, Copy, Debug, Default)]
pub struct Progress {
  /// The count of numbers written to `dst`.
  pub n_processed: usize,
  /// Whether the decompressor finished all compressed data relevant to
  /// the unit.
  /// For instance,
  /// [`PageDecompressor::read`][crate::wrapped::PageDecompressor::read]
  /// will return whether all data in the page was finished.
  pub finished: bool,
}
