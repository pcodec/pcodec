typedef enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  PcoCompressionError,
  PcoDecompressionError,
} PcoError;

/**
 * Configuration for compression, passed by the caller.
 *
 * Only `compression_level` and `paging_spec` are supported for now; other
 * fields can be added later without breaking the ABI.
 */
typedef struct PcoChunkConfig {
  /**
   * Compression level 0-12 (default 8).
   */
  unsigned int compression_level;
  /**
   * Maximum number of elements per page.
   * Set to 0 to use the library default (2^18 = 262144).
   */
  size_t max_page_n;
} PcoChunkConfig;

/**
 * Return the maximum possible byte size of a standalone file for `n`
 * elements of `dtype`.  Returns 0 for an invalid `dtype` or invalid
 * paging spec.
 *
 * This function is thread-safe and performs no heap allocation.
 */
size_t pco_standalone_guarantee_file_size(size_t n, unsigned char dtype);

/**
 * Compress `n` numbers of `dtype` from `nums` into the caller-owned buffer
 * `dst` (capacity `dst_cap` bytes).  On success `*n_written` is the number
 * of compressed bytes written.
 *
 * Thread-safe: the function is stateless and operates entirely on the
 * caller-supplied buffers.
 */
enum PcoError pco_standalone_simple_compress_into(const void *nums,
                                                  size_t n,
                                                  unsigned char dtype,
                                                  const struct PcoChunkConfig *config,
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
enum PcoError pco_standalone_simple_decompress_into(const void *compressed,
                                                    size_t compressed_len,
                                                    unsigned char dtype,
                                                    void *dst,
                                                    size_t dst_cap,
                                                    size_t *n_written);
