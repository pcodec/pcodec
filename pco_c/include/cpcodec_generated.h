typedef enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  /**
   * Generic compression failure, e.g. an IO error from the destination.
   */
  PcoCompressionError,
  /**
   * Generic decompression failure of a kind not covered below.
   */
  PcoDecompressionError,
  /**
   * The parameters provided were invalid, e.g. an unsupported compression
   * level.
   */
  PcoInvalidArgumentError,
  /**
   * The provided data is inconsistent or violates the pco format.
   */
  PcoCorruptionError,
  /**
   * The provided data ended before decompression finished.
   */
  PcoInsufficientDataError,
} PcoError;

/**
 * Configuration for compression, passed by the caller.
 *
 * Only `compression_level` and `paging_spec` are supported for now; other
 * fields can be added later without breaking the ABI.
 */
typedef struct PcoChunkConfig {
  /**
   * Compression level 0–12 (default 8).
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
 * Inspect a standalone file's header without decompressing it.
 *
 * On success, `*dtype` is set to the file's uniform number type byte (or 0
 * if the file does not declare a uniform type) and `*n_hint` is set to the
 * file's count hint: the total number of elements in the file if it was
 * recorded at compression time, or 0 if unknown.  Files written by
 * `pco_standalone_simple_compress_into` always record an exact count hint.
 *
 * Thread-safe: the function is stateless and operates entirely on the
 * caller-supplied buffers.
 */
enum PcoError pco_standalone_file_info(const void *compressed,
                                       size_t compressed_len,
                                       unsigned char *dtype,
                                       size_t *n_hint);

/**
 * Decompress `compressed_len` bytes from `compressed` into the caller-owned
 * buffer `dst` (capacity `dst_cap` *elements* of `dtype`), decompressing as
 * many elements as fit.
 *
 * Unlike `pco_standalone_simple_decompress_into`, an undersized `dst` is not
 * an error: on success `*n_written` is the number of elements written and
 * `*finished` is 1 if the entire file was decompressed, 0 if elements remain.
 *
 * Thread-safe: the function is stateless and operates entirely on the
 * caller-supplied buffers.
 */
enum PcoError pco_standalone_simple_decompress_partial_into(const void *compressed,
                                                            size_t compressed_len,
                                                            unsigned char dtype,
                                                            void *dst,
                                                            size_t dst_cap,
                                                            size_t *n_written,
                                                            unsigned char *finished);

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
