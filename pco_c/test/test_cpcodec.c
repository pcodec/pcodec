#include "../include/cpcodec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Caller-allocates API tests (thread-safe)
 * ------------------------------------------------------------------------- */

static int test_caller_alloc_api(void) {
  double input[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  int num_elems = sizeof(input) / sizeof(input[0]);
  int retcode = 0;

  printf("\n=== Caller-allocates API (thread-safe) ===\n");

  /* --- Compression ---------------------------------------------------- */
  size_t bound = pco_compress_bound(num_elems, PCO_TYPE_F64);
  if (bound == 0) {
    printf("FAIL: pco_compress_bound returned 0\n");
    return 1;
  }
  printf("Compression bound for %d f64s: %zu bytes\n", num_elems, bound);

  unsigned char *cbuf = (unsigned char *)malloc(bound);
  if (!cbuf) { printf("FAIL: malloc\n"); return 1; }

  size_t compressed_len = 0;
  enum PcoError res = pco_simple_compress_into(input, num_elems, PCO_TYPE_F64, 8,
                                               cbuf, bound, &compressed_len);
  if (res != PcoSuccess) {
    printf("FAIL: pco_simple_compress_into error %d\n", res);
    free(cbuf);
    return 1;
  }
  printf("Compressed %d f64s to %zu bytes\n", num_elems, compressed_len);

  /* --- Decompression -------------------------------------------------- */
  double *dbuf = (double *)malloc(num_elems * sizeof(double));
  if (!dbuf) { printf("FAIL: malloc\n"); free(cbuf); return 1; }

  size_t decompressed_n = 0;
  res = pco_simple_decompress_into(cbuf, compressed_len, PCO_TYPE_F64,
                                   dbuf, num_elems, &decompressed_n);
  if (res != PcoSuccess) {
    printf("FAIL: pco_simple_decompress_into error %d\n", res);
    retcode = 1;
    goto cleanup;
  }
  printf("Decompressed %zu f64s\n", decompressed_n);

  if ((int)decompressed_n != num_elems) {
    printf("FAIL: size mismatch (got %zu, want %d)\n", decompressed_n, num_elems);
    retcode = 1;
    goto cleanup;
  }
  for (int i = 0; i < num_elems; i++) {
    if (input[i] != dbuf[i]) {
      printf("FAIL: value mismatch at index %d (%.17g vs %.17g)\n",
             i, input[i], dbuf[i]);
      retcode = 1;
      goto cleanup;
    }
  }
  printf("Values match\n");

cleanup:
  free(dbuf);
  free(cbuf);
  return retcode;
}

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */

int main(void) {
  int rc = 0;
  rc |= test_caller_alloc_api();
  if (rc == 0)
    printf("\nAll tests passed.\n");
  else
    printf("\nSome tests FAILED.\n");
  return rc;
}
