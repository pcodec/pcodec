pub use dynamic::{LatentType, NumberType};
pub use split_latents::SplitLatents;

use crate::data_types::latent_priv::LatentPriv;
use crate::data_types::number_priv::NumberPriv;
use crate::describers::LatentDescriber;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, Mode};

mod dynamic;
pub(crate) mod float;
pub(crate) mod latent_priv;
pub(crate) mod number_priv;
pub(crate) mod signed;
mod split_latents;
pub(crate) mod unsigned;

pub(crate) type ModeAndLatents = (Mode, SplitLatents);

/// Trait for types that behave like unsigned integers.
///
/// Additionally, many types of Pco metadata are held in latents rather than
/// numbers directly; these can be interpreted via a
/// [`latent_describer`][`crate::data_types::Number::get_latent_describers`].
pub trait Latent: LatentPriv {}

impl<L: LatentPriv> Latent for L {}

/// Trait for types that Pco can compress and decompress.
pub trait Number: NumberPriv {
  fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber>;
}
