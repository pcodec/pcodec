[![Crates.io][crates-badge]][crates-url]

[crates-badge]: https://img.shields.io/crates/v/better_io.svg
[crates-url]: https://crates.io/crates/better_io

# Better IO

At present, this crate only supports `BetterBufRead` and `BetterBufReader`.
`BetterBufRead` is a new approach to buffered reading that I designed after
much thinking.
Though I don't think the ideas are original; they somewhat resemble
[this blog post](https://fgiesen.wordpress.com/2011/11/21/buffer-centric-io/)
and probably some older work too.
