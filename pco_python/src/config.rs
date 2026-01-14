use pco::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};
use pyo3::{pyclass, pymethods};

#[pyclass(name = "ModeSpec")]
#[derive(Clone, Default)]
pub struct PyModeSpec(ModeSpec);

/// Specifies how Pcodec should choose the mode.
#[pymethods]
impl PyModeSpec {
  /// :returns: a ModeSpec that automatically detects a good mode.
  #[staticmethod]
  fn auto() -> Self {
    Self(ModeSpec::Auto)
  }

  /// :returns: a ModeSpec that always uses the simplest mode.
  #[staticmethod]
  fn classic() -> Self {
    Self(ModeSpec::Classic)
  }

  /// :returns: a ModeSpec that tries to use FloatMult mode with the given base, if possible.
  #[staticmethod]
  fn try_float_mult(base: f64) -> Self {
    Self(ModeSpec::TryFloatMult(base))
  }

  /// :returns: a ModeSpec that tries to use FloatQuant mode with the given shift, if possible.
  #[staticmethod]
  fn try_float_quant(k: u32) -> Self {
    Self(ModeSpec::TryFloatQuant(k))
  }

  /// :returns: a ModeSpec that tries to use IntMult mode with the given base, if possible.
  #[staticmethod]
  fn try_int_mult(base: u64) -> Self {
    Self(ModeSpec::TryIntMult(base))
  }

  /// :returns: a ModeSpec that tries to use Dict mode, if possible.
  #[staticmethod]
  fn try_dict() -> Self {
    Self(ModeSpec::TryDict)
  }
}

#[pyclass(name = "DeltaSpec")]
#[derive(Clone, Default)]
pub struct PyDeltaSpec(DeltaSpec);

/// Specifies how Pcodec should choose the delta encoding.
#[pymethods]
impl PyDeltaSpec {
  /// :returns: a DeltaSpec that automatically detects a good delta encoding.
  #[staticmethod]
  fn auto() -> Self {
    Self(DeltaSpec::Auto)
  }

  /// :returns: a DeltaSpec that never does delta encoding.
  #[staticmethod]
  fn none() -> Self {
    Self(DeltaSpec::NoOp)
  }

  /// :returns: a DeltaSpec that tries to use the specified delta encoding order, if possible.
  #[staticmethod]
  fn try_consecutive(order: usize) -> Self {
    Self(DeltaSpec::TryConsecutive(order))
  }

  /// :returns: a DeltaSpec that tries to use delta lookbacks, if possible.
  #[staticmethod]
  fn try_lookback() -> Self {
    Self(DeltaSpec::TryLookback)
  }

  /// :returns: a DeltaSpec that tries to use 1D convolutions (equivalent to LPC) of the specified order, if possible.
  #[staticmethod]
  fn try_conv1(order: usize) -> Self {
    Self(DeltaSpec::TryConv1(order))
  }
}

#[pyclass(name = "PagingSpec")]
#[derive(Clone, Default)]
pub struct PyPagingSpec(PagingSpec);

/// Determines how pcodec splits a chunk into pages. In
/// standalone.simple_compress, this instead controls how pcodec splits a file
/// into chunks.
#[pymethods]
impl PyPagingSpec {
  /// :returns: a PagingSpec configuring a roughly count of numbers in each page.
  #[staticmethod]
  fn equal_pages_up_to(n: usize) -> Self {
    Self(PagingSpec::EqualPagesUpTo(n))
  }

  /// :returns: a PagingSpec with the exact, provided count of numbers in each page.
  #[staticmethod]
  fn exact_page_sizes(sizes: Vec<usize>) -> Self {
    Self(PagingSpec::Exact(sizes))
  }
}

#[pyclass(get_all, set_all, name = "ChunkConfig")]
pub struct PyChunkConfig {
  compression_level: usize,
  mode_spec: PyModeSpec,
  delta_spec: PyDeltaSpec,
  paging_spec: PyPagingSpec,
  enable_8_bit: bool,
}

#[pymethods]
impl PyChunkConfig {
  /// Creates a ChunkConfig.
  ///
  /// :param compression_level: a compression level from 0-12, where 12 takes
  ///   the longest and compresses the most.
  ///
  /// :param mode_spec: a ModeSpec to configure Pco's mode.
  ///
  /// :param delta_spec: a DeltaSpec to configure delta encoding.
  ///
  /// :param paging_spec: a PagingSpec describing how many numbers should
  ///   go into each page.
  ///
  /// :param enable_8_bit: whether to allow compression of 8-bit data types.
  ///
  /// :returns: A new ChunkConfig object.
  #[new]
  #[pyo3(signature = (
    compression_level=pco::DEFAULT_COMPRESSION_LEVEL,
    mode_spec=PyModeSpec::default(),
    delta_spec=PyDeltaSpec::default(),
    paging_spec=PyPagingSpec::default(),
    enable_8_bit=false,
  ))]
  fn new(
    compression_level: usize,
    mode_spec: PyModeSpec,
    delta_spec: PyDeltaSpec,
    paging_spec: PyPagingSpec,
    enable_8_bit: bool,
  ) -> Self {
    Self {
      compression_level,
      delta_spec,
      mode_spec,
      paging_spec,
      enable_8_bit,
    }
  }
}

impl From<&PyChunkConfig> for ChunkConfig {
  fn from(py_config: &PyChunkConfig) -> Self {
    ChunkConfig::default()
      .with_compression_level(py_config.compression_level)
      .with_delta_spec(py_config.delta_spec.0)
      .with_mode_spec(py_config.mode_spec.0)
      .with_paging_spec(py_config.paging_spec.0.clone())
      .with_enable_8_bit(py_config.enable_8_bit)
  }
}
