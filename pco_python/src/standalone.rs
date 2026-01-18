use numpy::{
  Element, IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyNone};
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};

use pco::data_types::{Number, NumberType};
use pco::standalone::FileDecompressor;
use pco::{match_number_enum, standalone, ChunkConfig};

use crate::utils::pco_err_to_py;
use crate::{utils, PyChunkConfig, PyProgress};

fn simple_compress_generic<'py, T: Number + Element>(
  py: Python<'py>,
  nums: &Bound<'_, PyArray1<T>>,
  config: &ChunkConfig,
) -> PyResult<Bound<'py, PyBytes>> {
  let nums = nums.readonly();
  let src = nums.as_slice()?;
  let compressed = py
    .detach(|| standalone::simple_compress(src, config))
    .map_err(pco_err_to_py)?;
  // TODO apparently all the places we use PyBytes::new() copy the data.
  // Maybe there's a zero-copy way to do this.
  Ok(PyBytes::new(py, &compressed))
}

fn simple_decompress_into_generic<T: Number + Element>(
  py: Python,
  src: &Bound<PyBytes>,
  dst: &Bound<PyArray1<T>>,
) -> PyResult<PyProgress> {
  let mut dst_rw = dst.readwrite();
  let dst = dst_rw.as_slice_mut()?;
  let src = src.as_bytes();
  let progress = py
    .detach(|| standalone::simple_decompress_into(src, dst))
    .map_err(pco_err_to_py)?;
  Ok(PyProgress::from(progress))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
  /// Compresses an array into a standalone format.
  ///
  /// :param nums: numpy array to compress. This must be 1D, contiguous, and
  ///   one of Pco's supported data types, e.g. float16, uint64.
  /// :param config: a ChunkConfig object containing compression level and
  ///   other settings.
  ///
  /// :returns: compressed bytes for an entire standalone file
  ///
  /// :raises: TypeError, RuntimeError
  #[pyfunction]
  fn simple_compress<'py>(
    py: Python<'py>,
    nums: &Bound<'_, PyUntypedArray>,
    config: &PyChunkConfig,
  ) -> PyResult<Bound<'py, PyBytes>> {
    let number_type = utils::number_type_from_numpy(py, &nums.dtype())?;
    match_number_enum!(
      number_type,
      NumberType<T> => {
        simple_compress_generic(py, utils::downcast_to_flat::<T>(nums)?, &config.clone().into())
      }
    )
  }
  m.add_function(wrap_pyfunction!(simple_compress, m)?)?;

  /// Decompresses pcodec compressed bytes into a pre-existing array.
  ///
  /// :param src: a bytes object a full standalone file of compressed data.
  /// :param dst: a numpy array to fill with the decompressed values. Must be
  ///   both 1D and contiguous.
  ///
  /// :returns: progress, an object with a count of elements written and
  ///   whether the compressed data was finished. If dst is shorter than the
  ///   numbers in compressed, fills dst and ignores the numbers that didn't
  ///   fit. If dst is longer, fills as much of dst as possible.
  ///
  /// :raises: TypeError, RuntimeError
  #[pyfunction]
  fn simple_decompress_into(
    py: Python,
    src: &Bound<PyBytes>,
    dst: &Bound<PyUntypedArray>,
  ) -> PyResult<PyProgress> {
    let number_type = utils::number_type_from_numpy(py, &dst.dtype())?;
    match_number_enum!(
      number_type,
      NumberType<T> => {
        simple_decompress_into_generic(py, src, utils::downcast_to_flat::<T>(dst)?)
      }
    )
  }
  m.add_function(wrap_pyfunction!(simple_decompress_into, m)?)?;

  /// Decompresses pcodec compressed bytes into a new Numpy array.
  ///
  /// :param src: a bytes object a full standalone file of compressed data.
  ///
  /// :returns: data, either a 1D numpy array of the decompressed values or, in
  ///   the event that there are no values and the data has no uniform data type
  ///   metadata, a None.
  ///
  /// :raises: TypeError, RuntimeError
  #[pyfunction]
  fn simple_decompress(py: Python, src: &Bound<PyBytes>) -> PyResult<Py<PyAny>> {
    let src = src.as_bytes();
    let (file_decompressor, src) = FileDecompressor::new(src).map_err(pco_err_to_py)?;
    let maybe_number_type = file_decompressor
      .peek_number_type_or_termination(src)
      .map_err(pco_err_to_py)?;
    match maybe_number_type {
      Some(number_type) => {
        match_number_enum!(
          number_type,
          NumberType<T> => {
            Ok(py
              .detach(|| file_decompressor.simple_decompress::<T>(src))
              .map_err(pco_err_to_py)?
              .into_pyarray(py)
              .into_pyobject(py)?
              .into())
          }
        )
      }
      None => Ok(PyNone::get(py).to_owned().into_pyobject(py)?.into()),
    }
  }
  m.add_function(wrap_pyfunction!(simple_decompress, m)?)?;

  Ok(())
}
