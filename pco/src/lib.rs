#![doc = include_str!("../README.md")]
//! # API Notes
//!
//! * In some places, Pco methods accept a destination (either `W: Write` or `&mut [T: NumberLike]`).
//! If Pco returns an error, it is possible both the destination and the struct
//! have been modified.
//! * Pco will always try to process all numbers, and it will fail if insufficient bytes are
//! available. For instance, during decompression Pco will try to fill the entire `&mut [T]`
//! passed in, returning an insufficient data error if the `&[u8]` passed in is not long enough.

#![allow(clippy::uninit_vec)]
#![deny(clippy::unused_unit)]
#![deny(dead_code)]

pub use bin::Bin;
pub use chunk_config::{ChunkConfig, ModeSpec, PagingSpec};
pub use chunk_meta::{ChunkLatentVarMeta, ChunkMeta};
pub use constants::{DEFAULT_COMPRESSION_LEVEL, DEFAULT_MAX_PAGE_N, FULL_BATCH_N};
pub use mode::Mode;
pub use progress::Progress;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct ReadmeDoctest;

pub mod data_types;
/// for inspecting certain types of Pco metadata
pub mod describers;
pub mod errors;
/// for compressing/decompressing .pco files
pub mod standalone;
/// for compressing/decompressing as part of an outer, wrapping format
pub mod wrapped;

mod ans;
mod bin;
mod bin_optimization;
mod bit_reader;
mod bit_writer;
mod bits;
mod chunk_config;
mod chunk_meta;
mod compression_intermediates;
mod compression_table;
mod constants;
mod delta;
mod float_mult_utils;
mod float_quant_utils;
mod format_version;
mod histograms;
mod int_mult_utils;
mod latent_batch_decompressor;
mod latent_batch_dissector;
mod mode;
mod page_meta;
mod progress;
mod read_write_uint;
mod sampling;
mod sort_utils;

#[cfg(test)]
mod tests;
