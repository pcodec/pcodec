pub use chunk_compressor::ChunkCompressor;
pub use chunk_decompressor::ChunkDecompressor;
pub use file_compressor::FileCompressor;
pub use file_decompressor::FileDecompressor;
pub use page_decompressor::PageDecompressor;
// Slightly unfortunate, but we expose this to standalone.
// Here's why: we want the wrapped PageDecompressor API to have a lifetime
// parameter for its ChunkDecompressor, so that users don't accidentally mix up
// PageDecompressorStates with the wrong chunks.
// Whereas for standalone we want to hide the distinction between chunks and
// pages, but storing a self-referencial standalone ChunkDecompressor would be
// absurdly annoying.
// The workaround is to interanlly expose the states without references to the
// original ChunkDecompressor.
pub(crate) use page_decompressor::PageDecompressorState;

mod chunk_compressor;
mod chunk_decompressor;
mod file_compressor;
mod file_decompressor;
/// Functions for guaranteed byte size upper bounds of components
/// like header and chunk metadata.
pub mod guarantee;
mod page_decompressor;
