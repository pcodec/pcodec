use crate::data_types::{Latent, Number, NumberPriv};
use crate::macros::{define_latent_enum, define_number_enum};

define_number_enum!(
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(u8)]
  pub NumberType = NUMBER_TYPE_BYTE
);

impl NumberType {
  #[doc(hidden)]
  #[deprecated(note = "use from_discriminant() instead")]
  pub fn from_descriminant(desc: u8) -> Option<Self> {
    Self::from_discriminant(desc)
  }
}

define_latent_enum!(
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub LatentType
);
