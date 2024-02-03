use numpy::PyArrayDyn;
use pco::data_types::CoreDataType;
use pco::{with_core_dtypes, ChunkConfig, FloatMultSpec, IntMultSpec, PagingSpec, Progress};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{pymodule, FromPyObject, PyModule, PyResult, Python};
use pyo3::{pyclass, pymethods, PyErr};

use pco::errors::PcoError;

pub mod standalone;
pub mod wrapped;

pub fn core_dtype_from_str(s: &str) -> PyResult<CoreDataType> {
  match s.to_uppercase().as_str() {
    "F32" => Ok(CoreDataType::F32),
    "F64" => Ok(CoreDataType::F64),
    "I32" => Ok(CoreDataType::I32),
    "I64" => Ok(CoreDataType::I64),
    "U32" => Ok(CoreDataType::U32),
    "U64" => Ok(CoreDataType::U64),
    _ => Err(PyRuntimeError::new_err(format!(
      "unknown data type: {}",
      s,
    ))),
  }
}

#[pyclass(get_all, name = "Progress")]
pub struct PyProgress {
  /// count of decompressed numbers.
  n_processed: usize,
  /// whether the compressed data was finished.
  finished: bool,
}

impl From<Progress> for PyProgress {
  fn from(progress: Progress) -> Self {
    Self {
      n_processed: progress.n_processed,
      finished: progress.finished,
    }
  }
}

pub fn pco_err_to_py(pco: PcoError) -> PyErr {
  PyRuntimeError::new_err(format!("pco error: {}", pco))
}

#[pyclass(get_all, set_all, name = "ChunkConfig")]
pub struct PyChunkConfig {
  compression_level: usize,
  delta_encoding_order: Option<usize>,
  int_mult_spec: String,
  float_mult_spec: String,
  max_page_n: usize,
}

#[pymethods]
impl PyChunkConfig {
  /// Creates a ChunkConfig.
  ///
  /// :param compression_level: a compression level from 0-12, where 12 takes
  /// the longest and compresses the most.
  /// :param delta_encoding_order: either a delta encoding level from 0-7 or
  /// None. If set to None, pcodec will try to infer the optimal delta encoding
  /// order.
  /// :param int_mult_spec: either 'enabled' or 'disabled'. If enabled, pcodec
  /// will consider using int mult mode, which can substantially improve
  /// compression ratio but decrease speed in some cases for integer types.
  /// :param float_mult_spec: either 'enabled' or 'disabled'. If enabled, pcodec
  /// will consider using float mult mode, which can substantially improve
  /// compression ratio but decrease speed in some cases for float types.
  /// :param max_page_n: the maximum number of values to encoder per pcodec
  /// page. If set too high or too low, pcodec's compression ratio may drop.
  ///
  /// :returns: A new ChunkConfig object.
  #[new]
  #[pyo3(signature = (
    compression_level=pco::DEFAULT_COMPRESSION_LEVEL,
    delta_encoding_order=None,
    int_mult_spec="enabled".to_string(),
    float_mult_spec="enabled".to_string(),
    max_page_n=262144,
  ))]
  fn new(
    compression_level: usize,
    delta_encoding_order: Option<usize>,
    int_mult_spec: String,
    float_mult_spec: String,
    max_page_n: usize,
  ) -> Self {
    Self {
      compression_level,
      delta_encoding_order,
      int_mult_spec,
      float_mult_spec,
      max_page_n,
    }
  }
}

impl TryFrom<&PyChunkConfig> for ChunkConfig {
  type Error = PyErr;

  fn try_from(py_config: &PyChunkConfig) -> Result<Self, Self::Error> {
    let int_mult_spec = match py_config.int_mult_spec.to_lowercase().as_str() {
      "enabled" => IntMultSpec::Enabled,
      "disabled" => IntMultSpec::Disabled,
      other => {
        return Err(PyRuntimeError::new_err(format!(
          "unknown int mult spec: {}",
          other
        )))
      }
    };
    let float_mult_spec = match py_config.float_mult_spec.to_lowercase().as_str() {
      "enabled" => FloatMultSpec::Enabled,
      "disabled" => FloatMultSpec::Disabled,
      other => {
        return Err(PyRuntimeError::new_err(format!(
          "unknown float mult spec: {}",
          other
        )))
      }
    };
    let res = ChunkConfig::default()
      .with_compression_level(py_config.compression_level)
      .with_delta_encoding_order(py_config.delta_encoding_order)
      .with_int_mult_spec(int_mult_spec)
      .with_float_mult_spec(float_mult_spec)
      .with_paging_spec(PagingSpec::EqualPagesUpTo(
        py_config.max_page_n,
      ));
    Ok(res)
  }
}

// The Numpy crate recommends using this type of enum to write functions that accept different Numpy dtypes
// https://github.com/PyO3/rust-numpy/blob/32740b33ec55ef0b7ebec726288665837722841d/examples/simple/src/lib.rs#L113
// The first dyn refers to dynamic dtype; the second to dynamic shape
#[derive(Debug, FromPyObject)]
pub enum DynTypedPyArrayDyn<'py> {
  F32(&'py PyArrayDyn<f32>),
  F64(&'py PyArrayDyn<f64>),
  I32(&'py PyArrayDyn<i32>),
  I64(&'py PyArrayDyn<i64>),
  U32(&'py PyArrayDyn<u32>),
  U64(&'py PyArrayDyn<u64>),
}

/// Pcodec is a codec for numerical sequences.
#[pymodule]
fn pcodec(py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add("__version__", env!("CARGO_PKG_VERSION"))?;
  m.add_class::<PyProgress>()?;
  m.add_class::<PyChunkConfig>()?;
  m.add(
    "DEFAULT_COMPRESSION_LEVEL",
    pco::DEFAULT_COMPRESSION_LEVEL,
  )?;

  // =========== STANDALONE ===========
  let standalone_module = PyModule::new(py, "standalone")?;
  standalone::register(py, standalone_module)?;
  m.add_submodule(standalone_module)?;

  // =========== WRAPPED ===========
  let wrapped_module = PyModule::new(py, "wrapped")?;
  wrapped::register(py, wrapped_module)?;
  m.add_submodule(wrapped_module)?;

  Ok(())
}
