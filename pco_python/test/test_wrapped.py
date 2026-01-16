import numpy as np
import pytest
from pcodec import ChunkConfig, PagingSpec
from pcodec.wrapped import FileCompressor, FileDecompressor

np.random.seed(12345)
all_dtypes = ("f2", "f4", "f8", "i2", "i4", "i8", "u2", "u4", "u8")


@pytest.mark.parametrize("dtype", all_dtypes)
def test_compress(dtype):
    data = np.random.uniform(0, 1000, size=[10]).astype(dtype)
    pco_number_type = dtype[0].upper() + str(int(dtype[1]) * 8)
    page_sizes = [6, 4]  # so there are 2 pages

    # compress
    fc = FileCompressor()
    header = fc.write_header()
    cc = fc.chunk_compressor(
        data,
        ChunkConfig(paging_spec=PagingSpec.exact_page_sizes(page_sizes)),
    )
    assert cc.n_per_page() == page_sizes
    chunk_meta = cc.write_meta()
    page0 = cc.write_page(0)
    page1 = cc.write_page(1)
    with pytest.raises(RuntimeError, match="page idx exceeds num pages"):
        cc.write_page(2)

    # decompress
    fd, n_bytes_read = FileDecompressor.new(header)
    assert n_bytes_read == len(header)
    # check that undershooting is fine
    _, n_bytes_read = FileDecompressor.new(header + b"foo")
    assert n_bytes_read == len(header)
    cd, n_bytes_read = fd.chunk_decompressor(chunk_meta, pco_number_type)
    assert n_bytes_read == len(chunk_meta)

    # page 1, which has elements 6-10
    dst1 = np.zeros(100).astype(dtype)
    progress, n_bytes_read = cd.read_page_into(page1, 4, dst1)
    np.testing.assert_array_equal(dst1[4:], np.zeros(96))
    np.testing.assert_array_equal(dst1[:4], data[6:])
    assert n_bytes_read == len(page1)

    # page 0, which has elements 0-6
    dst0 = np.zeros(6).astype(dtype)
    progress, n_bytes_read = cd.read_page_into(page0, 6, dst0)
    np.testing.assert_array_equal(dst0, data[:6])
    assert n_bytes_read == len(page0)
