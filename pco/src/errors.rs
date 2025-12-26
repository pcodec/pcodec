use std::error::Error;
use std::fmt::{Display, Formatter};
use std::{fmt, io};

/// The different kinds of errors the library can return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
  /// `Corruption` errors occur during decompression, indicating the
  /// provided data is inconsistent or violates the pco format.
  /// These may arise when the decompressor version is too old.
  /// They may also apply to cases where standalone files were read using the
  /// wrapped API, or vice versa.
  Corruption,
  /// `InsufficientData` errors occur during decompression, indicating
  /// the decompressor reached the end of the provided data before finishing.
  InsufficientData,
  /// `InvalidArgument` errors usually occur during compression, indicating
  /// the parameters provided to a function were invalid.
  InvalidArgument,
  /// `Io` errors are propagated from `Read` or `Write`
  /// implementations passed to pco.
  Io(io::ErrorKind),
}

/// The error type used in results for all `pco` functionality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcoError {
  pub kind: ErrorKind,
  pub message: String,
}

impl PcoError {
  pub(crate) fn new<S: AsRef<str>>(kind: ErrorKind, message: S) -> Self {
    PcoError {
      kind,
      message: message.as_ref().to_string(),
    }
  }

  pub(crate) fn corruption<S: AsRef<str>>(message: S) -> Self {
    Self::new(ErrorKind::Corruption, message)
  }

  pub(crate) fn insufficient_data<S: AsRef<str>>(message: S) -> Self {
    Self::new(ErrorKind::InsufficientData, message)
  }

  pub(crate) fn invalid_argument<S: AsRef<str>>(message: S) -> Self {
    Self::new(ErrorKind::InvalidArgument, message)
  }
}

impl Display for PcoError {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "pco {:?} error: {}",
      self.kind, &self.message
    )
  }
}

impl From<io::Error> for PcoError {
  fn from(err: io::Error) -> Self {
    PcoError {
      kind: ErrorKind::Io(err.kind()),
      message: format!("{}", err),
    }
  }
}

impl Error for PcoError {}

pub type PcoResult<T> = Result<T, PcoError>;
