import numpy as np
import pytest
from pcodec import (
    ChunkConfig,
    DeltaSpec,
    ModeSpec,
    PagingSpec,
    standalone,
)
from pathlib import Path

np.random.seed(12345)
all_lengths = (
    0,
    900,
)
all_dtypes = ("f2", "f4", "f8", "i2", "i4", "i8", "u2", "u4", "u8")


@pytest.mark.parametrize("length", all_lengths)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_round_trip_decompress_into(length, dtype):
    data = np.random.uniform(0, 1000, size=length).astype(dtype)
    compressed = standalone.simple_compress(data, ChunkConfig())

    # decompress exactly
    out = np.empty_like(data)
    progress = standalone.simple_decompress_into(compressed, out)
    np.testing.assert_array_equal(data, out)
    assert progress.n_processed == data.size
    assert progress.finished


@pytest.mark.parametrize("length", all_lengths)
@pytest.mark.parametrize("dtype", all_dtypes)
def test_round_trip_simple_decompress(length, dtype):
    data = np.random.uniform(0, 1000, size=length).astype(dtype)
    compressed = standalone.simple_compress(
        data, ChunkConfig(paging_spec=PagingSpec.equal_pages_up_to(300))
    )
    out = standalone.simple_decompress(compressed)
    np.testing.assert_array_equal(data, out)


def test_inexact_decompression():
    data = np.random.uniform(size=300)
    compressed = standalone.simple_compress(data, ChunkConfig())

    # decompress partially
    out = np.zeros(3)
    progress = standalone.simple_decompress_into(compressed, out)
    np.testing.assert_array_equal(out, data[:3])
    assert progress.n_processed == 3
    assert not progress.finished

    # decompress with room to spare
    out = np.zeros(600)
    progress = standalone.simple_decompress_into(compressed, out)
    np.testing.assert_array_equal(out[:300], data)
    np.testing.assert_array_equal(out[300:], np.zeros(300))
    assert progress.n_processed == 300
    assert progress.finished


def test_simple_decompress_into_errors():
    """Test possible error states for standalone.simple_decompress_into"""
    data = np.random.uniform(size=100).astype(np.float32)
    compressed = standalone.simple_compress(data, ChunkConfig())

    out = np.zeros(100).astype(np.float64)
    with pytest.raises(RuntimeError, match="does not match chunk's number type"):
        standalone.simple_decompress_into(compressed, out)


def test_simple_decompress_errors():
    """Test possible error states for standalone.simple_decompress"""
    test_file_path = Path(__file__).parent / "../../pco/assets/v0_4_5_uniform_type.pco"
    with open(test_file_path, "rb") as f:
        compressed = bytearray(f.read())

    # byte 5 is the uniform number type
    # byte 8 is the first chunk's number type
    truncated = compressed[:8]
    with pytest.raises(RuntimeError, match="InsufficientData"):
        standalone.simple_decompress(bytes(truncated))

    compressed[8] = 99
    with pytest.raises(
        RuntimeError,
        match="chunk's number type of 99 does not match file's uniform number type of U32",
    ):
        standalone.simple_decompress(bytes(compressed))

    # new behavior when using uniform type: returns an empty array.
    # this happens if the user passed in a file with no chunks.
    compressed[8] = 0
    np.testing.assert_array_equal(
        standalone.simple_decompress(bytes(compressed)), np.array([], dtype=np.uint32)
    )

    # files not using uniform dtypes store no dtype and return None instead.
    compressed[5] = 0
    assert standalone.simple_decompress(bytes(compressed)) is None


def test_compression_options():
    data = np.random.normal(size=100).astype(np.float32)
    default_size = len(standalone.simple_compress(data, ChunkConfig()))

    # this is mostly just to check that there is no error, but these settings
    # should give worse compression than the defaults
    for delta_spec in [DeltaSpec.try_consecutive(1), DeltaSpec.try_lookback()]:
        compressed = standalone.simple_compress(
            data,
            ChunkConfig(
                compression_level=0,
                delta_spec=delta_spec,
                mode_spec=ModeSpec.classic(),
                paging_spec=PagingSpec.equal_pages_up_to(77),
            ),
        )
        assert len(compressed) > default_size


@pytest.mark.parametrize(
    "mode_spec", [ModeSpec.auto(), ModeSpec.classic(), ModeSpec.try_int_mult(10)]
)
def test_compression_int_mode_spec_options(mode_spec):
    data = (np.random.normal(size=100) * 1000).astype(np.int32)

    # check for errors
    compressed = standalone.simple_compress(
        data,
        ChunkConfig(mode_spec=mode_spec),
    )

    out = standalone.simple_decompress(compressed)

    # check that the decompressed data is correct
    np.testing.assert_array_equal(data, out)


@pytest.mark.parametrize(
    "mode_spec",
    [
        ModeSpec.auto(),
        ModeSpec.classic(),
        ModeSpec.try_float_mult(10.0),
        ModeSpec.try_float_quant(4),
    ],
)
def test_compression_float_mode_spec_options(mode_spec):
    data = (np.random.normal(size=100) * 1000).astype(np.int32) * np.pi

    # check for errors
    compressed = standalone.simple_compress(
        data,
        ChunkConfig(mode_spec=mode_spec),
    )

    out = standalone.simple_decompress(compressed)

    # check that the decompressed data is correct
    np.testing.assert_array_equal(data, out)


def test_decompress_without_n_hint():
    # old files didn't have n_hint
    with open(Path(__file__).parent / "../../pco/assets/v0_0_0_classic.pco", "rb") as f:
        compressed = f.read()

    assert len(standalone.simple_decompress(compressed)) == 2000
