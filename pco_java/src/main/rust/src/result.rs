use jni::errors::Error as JniError;
use pco::errors::{ErrorKind as PcoErrorKind, PcoError};

#[derive(Clone, Debug)]
pub enum ExceptionKind {
  InvalidArgument,
  Io,
  Runtime,
}

#[derive(Clone, Debug)]
pub struct Exception {
  pub kind: ExceptionKind,
  pub msg: String,
}

pub type Result<T> = std::result::Result<T, Exception>;

impl From<PcoError> for Exception {
  fn from(value: PcoError) -> Self {
    let msg = format!("{}", value);
    let kind = match value.kind {
      PcoErrorKind::Io(_) => ExceptionKind::Io,
      PcoErrorKind::InvalidArgument => ExceptionKind::InvalidArgument,
      _ => ExceptionKind::Runtime,
    };
    Exception { kind, msg }
  }
}

impl From<JniError> for Exception {
  fn from(value: JniError) -> Self {
    Exception {
      kind: ExceptionKind::Runtime,
      msg: format!("{}", value),
    }
  }
}
