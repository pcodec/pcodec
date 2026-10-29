use anyhow::Result;
use clap::Parser;

use crate::opt::{Opt, OptWrapper};

mod arrow_handlers;
mod bench;
mod chunk_config_opt;
mod compress;
mod decompress;
mod dtypes;
mod input;
mod inspect;
pub mod num_vec;
mod opt;
mod parse;
mod utils;

fn main() -> Result<()> {
  let opt = OptWrapper::parse().opt;
  match opt {
    Opt::Bench(bench_opt) => bench::bench(bench_opt),
    Opt::Compress(compress_opt) => compress::compress(compress_opt),
    Opt::Decompress(decompress_opt) => decompress::decompress(decompress_opt),
    Opt::Inspect(inspect_opt) => inspect::inspect(inspect_opt),
  }
}
