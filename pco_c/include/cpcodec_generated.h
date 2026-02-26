typedef enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  PcoCompressionError,
  PcoDecompressionError,
} PcoError;

/**
 * Return an upper bound on the number of bytes required to compress `n`
 * elements of `dtype`.  Returns 0 for an invalid `dtype`.
 *
 * This function is thread-safe and performs no heap allocation.
 */
size_t pco_compress_bound(size_t n, unsigned char dtype);

/**
 * Compress `n` numbers of `dtype` from `nums` into the caller-owned buffer
 * `dst` (capacity `dst_cap` bytes).  On success `*n_written` is the number
 * of compressed bytes written.
 *
 * Thread-safe: the function is stateless and operates entirely on the
 * caller-supplied buffers.
 */
enum PcoError pco_simple_compress_into(const void *nums,
                                       size_t n,
                                       unsigned char dtype,
                                       unsigned int level,
                                       void *dst,
                                       size_t dst_cap,
                                       size_t *n_written);

/**
 * Decompress `compressed_len` bytes from `compressed` into the caller-owned
 * buffer `dst` (capacity `dst_cap` *elements* of `dtype`).  On success
 * `*n_written` is the number of elements written.
 *
 * Thread-safe: the function is stateless and operates entirely on the
 * caller-supplied buffers.
 */
enum PcoError pco_simple_decompress_into(const void *compressed,
                                         size_t compressed_len,
                                         unsigned char dtype,
                                         void *dst,
                                         size_t dst_cap,
                                         size_t *n_written);
