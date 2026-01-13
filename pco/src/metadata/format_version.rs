use std::fmt::Display;
use std::io::Write;

use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::errors::{PcoError, PcoResult};

/// The format version of pco used to compress a file.
///
/// During compression, this gets stored in the file.
/// Version can affect the encoding of the rest of the file, so older versions
/// of pco might return corruption errors when running on data compressed
/// by newer versions.
/// Pco will attempt to decompress if it's possible the library can read the
/// file (the library supports the file's major version), even if it's possible
/// the library will fail later in the file (which can happen if the library
/// doesn't yet support the file's minor version).
///
/// You will not need to manually instantiate this.
/// However, in some circumstances you may want to inspect this during
/// decompression.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FormatVersion {
  pub major: u8,
  pub minor: u8,
}

/// The default FormatVersion is used when compressing files.
impl Default for FormatVersion {
  fn default() -> Self {
    Self { major: 4, minor: 1 }
  }
}

impl Display for FormatVersion {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}.{}", self.major, self.minor)
  }
}

impl FormatVersion {
  pub(crate) const MAX_ENCODED_SIZE: usize = 2;

  /// Returns the max format version that can definitely be decompressed by the
  /// current library version.
  pub fn max_supported() -> Self {
    Self { major: 4, minor: 1 }
  }

  /// Returns whether this format version can definitely be read by the
  /// current library version.
  pub fn can_definitely_be_decompressed(&self) -> bool {
    Self::max_supported() >= *self
  }

  /// Returns whether this format version might be readable by the current
  /// library version (depending on whether new format additions have been used
  /// later in the file).
  pub fn can_maybe_be_decompressed(&self) -> bool {
    Self::max_supported().major >= self.major
  }

  pub(crate) fn read_from(reader: &mut BitReader) -> PcoResult<Self> {
    let major = reader.read_aligned_bytes(1)?[0];
    let minor = if major >= 4 {
      reader.read_aligned_bytes(1)?[0]
    } else {
      0
    };
    let file_version = FormatVersion { major, minor };

    if !file_version.can_maybe_be_decompressed() {
      return Err(PcoError::corruption(format!(
        "File's format version ({}) definitely cannot be decompressed by this \
        library version; its major version is less than the max supported {}. \
        Consider upgrading Pco.",
        file_version,
        FormatVersion::default(),
      )));
    }

    Ok(file_version)
  }

  pub(crate) fn write_to<W: Write>(&self, writer: &mut BitWriter<W>) -> PcoResult<usize> {
    assert!(self.major >= 4);
    writer.write_aligned_bytes(&[self.major, self.minor])?;
    Ok(2)
  }

  pub(crate) fn used_old_gcds(&self) -> bool {
    self.major == 0
  }

  pub(crate) fn supports_delta_variants(&self) -> bool {
    self.major >= 3
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn max_supported_exceeds_default() {
    assert!(FormatVersion::max_supported() >= FormatVersion::default());
  }

  #[test]
  fn ordering_properties() {
    let a = FormatVersion { major: 1, minor: 5 };
    let b = FormatVersion { major: 2, minor: 4 };
    let c = FormatVersion { major: 2, minor: 6 };

    assert!(a < b);
    assert!(b < c);
  }

  #[test]
  fn can_be_decompressed() {
    let past_version = FormatVersion {
      major: 0,
      minor: 50,
    };
    assert!(past_version.can_maybe_be_decompressed());
    assert!(past_version.can_definitely_be_decompressed());

    let uncertain_version = FormatVersion {
      major: FormatVersion::max_supported().major,
      minor: 200,
    };
    assert!(uncertain_version.can_maybe_be_decompressed());
    assert!(!uncertain_version.can_definitely_be_decompressed());

    let future_version = FormatVersion {
      major: 200,
      minor: 0,
    };
    assert!(!future_version.can_maybe_be_decompressed());
    assert!(!future_version.can_definitely_be_decompressed());
  }
}
