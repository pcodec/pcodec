use std::io::Write;

use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::CURRENT_FORMAT_VERSION;
use crate::errors::{PcoError, PcoResult};

/// The version of pco used to compress a file.
///
/// During compression, this gets stored in the file.
/// Version can affect the encoding of the rest of the file, so older versions
/// of pco might return corruption errors when running on data compressed
/// by newer versions.
///
/// You will not need to manually instantiate this.
/// However, in some circumstances you may want to inspect this during
/// decompression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatVersion(pub u8);

impl Default for FormatVersion {
  fn default() -> Self {
    Self(CURRENT_FORMAT_VERSION)
  }
}

impl FormatVersion {
  pub(crate) const MAX_ENCODED_SIZE: usize = 1;

  pub(crate) fn read_from(reader: &mut BitReader) -> PcoResult<Self> {
    let version = reader.read_aligned_bytes(1)?[0];
    if version > CURRENT_FORMAT_VERSION {
      return Err(PcoError::corruption(format!(
        "file's format version ({}) exceeds max supported ({}); consider upgrading pco",
        version, CURRENT_FORMAT_VERSION,
      )));
    }

    Ok(Self(version))
  }

  pub(crate) fn write_to<W: Write>(&self, writer: &mut BitWriter<W>) -> PcoResult<usize> {
    // in the future, we may want to allow the user to encode with their choice of a recent version
    writer.write_aligned_bytes(&[self.0])?;
    Ok(1)
  }

  pub(crate) fn used_old_gcds(&self) -> bool {
    self.0 == 0
  }

  pub(crate) fn supports_delta_variants(&self) -> bool {
    self.0 >= 3
  }
}
