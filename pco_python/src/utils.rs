use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pco::data_types::NumberType;
use pco::errors::PcoError;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::{Bound, PyErr, PyResult, Python};

pub fn core_dtype_from_str(s: &str) -> PyResult<NumberType> {
  match s.to_uppercase().as_str() {
    "F16" => Ok(NumberType::F16),
    "F32" => Ok(NumberType::F32),
    "F64" => Ok(NumberType::F64),
    "I16" => Ok(NumberType::I16),
    "I32" => Ok(NumberType::I32),
    "I64" => Ok(NumberType::I64),
    "U16" => Ok(NumberType::U16),
    "U32" => Ok(NumberType::U32),
    "U64" => Ok(NumberType::U64),
    _ => Err(PyRuntimeError::new_err(format!(
      "unknown data type: {}",
      s,
    ))),
  }
}

pub fn number_type_from_numpy(py: Python, dtype: &Bound<PyArrayDescr>) -> PyResult<NumberType> {
  let res = if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
    NumberType::U16
  } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
    NumberType::U32
  } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
    NumberType::U64
  } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
    NumberType::I16
  } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
    NumberType::I32
  } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
    NumberType::I64
  } else if dtype.is_equiv_to(&numpy::dtype::<half::f16>(py)) {
    NumberType::F16
  } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
    NumberType::F32
  } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
    NumberType::F64
  } else {
    return Err(PyTypeError::new_err(format!(
      "Unsupported data type: {:?}",
      dtype
    )));
  };
  Ok(res)
}

pub fn pco_err_to_py(pco: PcoError) -> PyErr {
  PyRuntimeError::new_err(format!("pco error: {}", pco))
}
