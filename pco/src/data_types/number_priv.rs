use std::fmt::{Debug, Display};

use crate::data_types::latent_priv::LatentPriv;
use crate::data_types::ModeAndLatents;
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::Mode;
use crate::ChunkConfig;

pub trait NumberPriv: Copy + Debug + Display + Default + PartialEq + Send + Sync + 'static {
  // To choose a header byte for a new data type, review all header bytes in
  // the library and pick the next higher byte.
  // `pco` data type implementation.
  /// A number from 1-255 that corresponds to the number's data type.
  ///
  /// Each `Number` implementation should have a different `NUMBER_TYPE_BYTE`.
  /// This byte gets written into the file's header during compression, and
  /// if the wrong header byte shows up during decompression, the decompressor
  /// will return an error.
  const NUMBER_TYPE_BYTE: u8;

  /// The latent this type can convert between to do bitwise logic and such.
  type L: LatentPriv;

  fn mode_is_valid(mode: &Mode) -> bool;
  /// Breaks the numbers into latent variables for better compression.
  ///
  /// Returns
  /// * mode: the [`Mode`] that will be stored alongside the data
  ///   for decompression
  /// * latents: a primary and optionally secondary latent variable, each of
  ///   which contains a latent per num in `nums`. Primary must be of the same
  ///   latent type as T.
  fn choose_mode_and_split_latents(
    nums: &[Self],
    config: &ChunkConfig,
  ) -> PcoResult<ModeAndLatents>;

  fn from_latent_ordered(l: Self::L) -> Self;
  fn to_latent_ordered(self) -> Self::L;
  fn join_latents(
    mode: &Mode,
    primary: DynLatentSlice,
    secondary: Option<DynLatentSlice>,
    dst: &mut [Self],
  ) -> PcoResult<()>;
}
