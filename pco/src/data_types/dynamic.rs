use crate::data_types::{Latent, Number, NumberPriv};
use crate::macros::{define_latent_enum, define_number_enum};

define_number_enum!(
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(u8)]
  pub NumberType = NUMBER_TYPE_BYTE
);

define_latent_enum!(
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub LatentType
);
