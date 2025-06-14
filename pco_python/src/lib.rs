#![allow(clippy::uninit_vec)]
#![deny(clippy::unused_unit)]
#![deny(dead_code)]

use crate::config::{PyChunkConfig, PyDeltaSpec, PyModeSpec, PyPagingSpec};
use crate::progress::PyProgress;
use pyo3::prelude::*;
use pyo3::{py_run, Bound};

mod config;
mod progress;
pub mod standalone;
mod utils;
pub mod wrapped;

/// Pcodec is a codec for numerical sequences.
#[pymodule]
fn pcodec(m: &Bound<PyModule>) -> PyResult<()> {
  let py = m.py();
  m.add("__version__", env!("CARGO_PKG_VERSION"))?;
  m.add_class::<PyProgress>()?;
  m.add_class::<PyModeSpec>()?;
  m.add_class::<PyDeltaSpec>()?;
  m.add_class::<PyPagingSpec>()?;
  m.add_class::<PyChunkConfig>()?;
  m.add(
    "DEFAULT_COMPRESSION_LEVEL",
    pco::DEFAULT_COMPRESSION_LEVEL,
  )?;

  // =========== STANDALONE ===========
  let standalone_module = PyModule::new(py, "pcodec.standalone")?;
  standalone::register(&standalone_module)?;
  // hackery from https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
  // to make modules work nicely
  py_run!(
    py,
    standalone_module,
    "import sys; sys.modules['pcodec.standalone'] = standalone_module"
  );
  m.add_submodule(&standalone_module)?;

  // =========== WRAPPED ===========
  let wrapped_module = PyModule::new(py, "pcodec.wrapped")?;
  wrapped::register(&wrapped_module)?;
  py_run!(
    py,
    wrapped_module,
    "import sys; sys.modules['pcodec.wrapped'] = wrapped_module"
  );
  m.add_submodule(&wrapped_module)?;

  Ok(())
}
