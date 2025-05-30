use std::convert::TryInto;

use numpy::{Element, PyArray1, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};

use pco::data_types::{Number, NumberType};
use pco::wrapped::{ChunkCompressor, FileCompressor};
use pco::{match_number_enum, ChunkConfig};

use crate::utils::pco_err_to_py;
use crate::{utils, PyChunkConfig};

/// The top-level object for creating wrapped pcodec files.
/// Has a default constructor.
#[pyclass(name = "FileCompressor")]
struct PyFc {
  inner: FileCompressor,
}

// can't pass inner directly since pyo3 only supports unit variant enums
/// Holds metadata about a chunk and supports compressing one page at a time.
#[pyclass(name = "ChunkCompressor")]
struct PyCc(ChunkCompressor);

impl PyFc {
  fn chunk_compressor_generic<T: Number + Element>(
    &self,
    py: Python,
    arr: &Bound<PyArray1<T>>,
    config: &ChunkConfig,
  ) -> PyResult<ChunkCompressor> {
    let arr_ro = arr.readonly();
    let src = arr_ro.as_slice()?;
    py.allow_threads(|| self.inner.chunk_compressor(src, config))
      .map_err(pco_err_to_py)
  }
}

#[pymethods]
impl PyFc {
  /// :returns: a new FileCompressor.
  #[new]
  pub fn new() -> PyFc {
    PyFc {
      inner: FileCompressor::default(),
    }
  }

  /// :returns: a bytes object containing the encoded header
  ///
  /// :raises: TypeError, RuntimeError
  fn write_header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    let mut res = Vec::new();
    self.inner.write_header(&mut res).map_err(pco_err_to_py)?;
    Ok(PyBytes::new(py, &res))
  }

  /// Create a chunk compressor, computing the chunk metadata necessary to
  /// compress the provided nums.
  ///
  /// This does the bulk of the work of compression.
  ///
  /// :param nums: numpy array to compress. This may have any shape.
  ///   However, it must be contiguous, and only the following data types are
  ///   supported: float16, float32, float64, int16, int32, int64, uint16, uint32, uint64.
  /// :param config: a ChunkConfig object containing compression level and
  ///   other settings.
  ///
  /// :returns: a ChunkCompressor
  ///
  /// :raises: TypeError, RuntimeError
  fn chunk_compressor(
    &self,
    py: Python,
    nums: Bound<PyUntypedArray>,
    config: &PyChunkConfig,
  ) -> PyResult<PyCc> {
    let config = config.try_into()?;
    let number_type = utils::number_type_from_numpy(py, &nums.dtype())?;
    match_number_enum!(
      number_type,
      NumberType<T> => {
        let cc = self.chunk_compressor_generic::<T>(py, nums.downcast::<PyArray1<T>>()?, &config)?;
        Ok(PyCc(cc))
      }
    )
  }
}

#[pymethods]
impl PyCc {
  /// :returns: a bytes object containing the encoded chunk metadata.
  ///
  /// :raises: TypeError, RuntimeError
  fn write_chunk_meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    let mut res = Vec::new();
    self.0.write_chunk_meta(&mut res).map_err(pco_err_to_py)?;
    Ok(PyBytes::new(py, &res))
  }

  /// :returns: a list containing the count of numbers in each page.
  fn n_per_page(&self) -> Vec<usize> {
    self.0.n_per_page()
  }

  /// :param page_idx: an int for which page you want to write.
  ///
  /// :returns: a bytes object containing the encoded page.
  ///
  /// :raises: TypeError, RuntimeError
  fn write_page<'py>(&self, py: Python<'py>, page_idx: usize) -> PyResult<Bound<'py, PyBytes>> {
    let mut res = Vec::new();
    py.allow_threads(|| self.0.write_page(page_idx, &mut res))
      .map_err(pco_err_to_py)?;
    Ok(PyBytes::new(py, &res))
  }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
  m.add_class::<PyFc>()?;
  m.add_class::<PyCc>()?;

  Ok(())
}
