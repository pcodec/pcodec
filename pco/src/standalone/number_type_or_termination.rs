use NumberTypeOrTermination::*;

use crate::data_types::NumberType;
use crate::standalone::constants::MAGIC_TERMINATION_BYTE;

/// A value representing either a standalone chunk's data type or a termination
/// byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberTypeOrTermination {
  /// Termination indicates that the file is ending. There are no chunks after
  /// this point.
  Termination,
  /// This indicates that there is a chunk, and its data type is the given
  /// known value.
  Known(NumberType),
  // TODO in 1.0 since it seems decided we aren't implementing custom data
  // types, it's unclear if Unknown adds anything over returning a
  // corruption error?
  /// This indicates that there is a chunk, but its data type is not part of
  /// pco's core implementation. It may be possible to decode it with custom
  /// data types.
  Unknown(u8),
}

impl From<NumberTypeOrTermination> for u8 {
  fn from(type_or_termination: NumberTypeOrTermination) -> u8 {
    match type_or_termination {
      Termination => MAGIC_TERMINATION_BYTE,
      Known(core) => core as u8,
      Unknown(byte) => byte,
    }
  }
}

impl From<u8> for NumberTypeOrTermination {
  fn from(byte: u8) -> Self {
    if byte == MAGIC_TERMINATION_BYTE {
      Termination
    } else if let Some(core) = NumberType::from_descriminant(byte) {
      Known(core)
    } else {
      Unknown(byte)
    }
  }
}
