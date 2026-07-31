# Seed corpora

`fuzz/corpus/` is the *working* corpus: cargo-fuzz writes into it on every run
and it is gitignored. This directory is the *seed* corpus that survives in the
repository, so a fresh clone does not start fuzzing from an empty set — reaching
the interesting states again costs CPU-minutes that nobody should have to pay
twice.

One reproducible `tar.zst` per target rather than loose files, to keep the tree
(and any diff of it) readable:

| target | files | uncompressed | archive |
| --- | --- | --- | --- |
| `decompress_arbitrary` | 23 | ~96 KB | 949 B |
| `decompress_corrupt` | 1377 | ~1.1 MB | 78 KB |
| `roundtrip` | 55 | ~224 KB | 2.3 KB |

## Use

```sh
cd fuzz
mkdir -p corpus
for t in decompress_arbitrary decompress_corrupt roundtrip; do
  tar -C corpus -xf "seeds/$t.tar.zst"
done
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

Minimized (`cargo fuzz cmin`) from a session of the runs recorded in
`../FINDINGS.md`, i.e. cargo-fuzz's default build mode (opt-level 3 **plus**
debug-assertions and overflow-checks). Coverage after minimization, as reported
by the merge step: 489 edges for `decompress_arbitrary`, 3485 for
`decompress_corrupt`, 2857 for `roundtrip`.

The gap between the first and the second is the point of `decompress_corrupt`:
`decompress_arbitrary` feeds unstructured bytes and almost never gets past the
magic header, while `decompress_corrupt` builds a valid file and then mutates
it, so nearly all of its inputs reach real decode paths. Seeds are therefore
worth much more to the arbitrary target than its own runs are.
