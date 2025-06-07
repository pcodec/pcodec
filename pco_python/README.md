<div style="text-align:center">
  <img alt="Pco logo: a pico-scale, compressed version of the Pyramid of Khafre in the palm of your hand" src="https://raw.githubusercontent.com/pcodec/pcodec/cac902e714077426d915f4fc397508b187c72380/images/logo.svg" width="160px">
</div>

[![pypi.org][pypi-badge]][pypi-url]

[pypi-badge]: https://img.shields.io/pypi/v/pcodec.svg

[pypi-url]: https://pypi.org/project/pcodec/

# Pcodec Python API

Pcodec is a codec for numerical sequences. Example usage:

```python
>>> from pcodec import standalone, ChunkConfig
>>> import numpy as np
>>> 
>>> np.random.seed(0)
>>> nums = np.random.normal(size=1000000)
>>> 
>>> # compress
>>> compressed = standalone.simple_compress(nums, ChunkConfig())
>>> print(f'compressed to {len(compressed)} bytes')
compressed to 6946278 bytes
>>> 
>>> # decompress
>>> recovered = standalone.simple_decompress(compressed)
>>> 
>>> np.testing.assert_array_equal(recovered, nums)

```

For pcodec's uses, design, and benchmarks, [see the main repo](https://github.com/pcodec/pcodec).

Documentation is served
[on readthedocs](https://pcodec.readthedocs.io/en/latest/).
It's sometimes incomplete, but the types and functions largely mirror those of
the better-documented Rust API.
