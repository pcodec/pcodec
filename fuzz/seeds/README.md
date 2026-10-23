# Seed corpora

`fuzz/corpus/` is the *working* corpus: cargo-fuzz writes into it on every run
and it is gitignored. This directory is the *seed* corpus that survives in the
repository, so a fresh clone does not start fuzzing from an empty set — reaching
the interesting states again costs CPU-minutes that nobody should have to pay
twice.

One reproducible `tar.zst` per target rather than loose files, to keep the tree
(and any diff of it) readable:

| target | files | archive | edges |
| --- | --- | --- | --- |
| `decompress_arbitrary` | 23 | 949 B | 489 |
| `decompress_corrupt` | 1377 | 78 KB | 3485 |
| `roundtrip` | 55 | 2.3 KB | 2857 |
| `c_api_roundtrip` | 2999 | 144 KB | 8650 |
| `c_api_decompress` | 978 | 53 KB | 3896 |

## Use

```sh
cd fuzz
mkdir -p corpus
for f in seeds/*.tar.zst; do tar -C corpus -xf "$f"; done
cargo fuzz run decompress_corrupt          # picks up corpus/decompress_corrupt
```

## Refresh after a fuzzing session

`cmin` first — the working corpus grows with inputs that add nothing, and only
the coverage-distinct ones are worth keeping:

```sh
cargo fuzz cmin decompress_corrupt
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='<a fixed date>' \
    -C corpus -cf - decompress_corrupt \
  | zstd -19 -q -o seeds/decompress_corrupt.tar.zst -f
```

The `--sort`/`--owner`/`--mtime` flags are what make the archive byte-stable, so
re-archiving an unchanged corpus produces no diff.

## Provenance

Minimized (`cargo fuzz cmin`) from the runs recorded in `../FINDINGS.md`, i.e.
cargo-fuzz's default build mode (opt-level 3 **plus** debug-assertions and
overflow-checks). The edge counts in the table are what the merge step reported
after minimization.

The gap between the first two rows is the point of `decompress_corrupt`:
`decompress_arbitrary` feeds unstructured bytes and almost never gets past the
magic header, while `decompress_corrupt` builds a valid file and then mutates
it, so nearly all of its inputs reach real decode paths. Seeds are therefore
worth much more to the arbitrary target than its own runs are.
