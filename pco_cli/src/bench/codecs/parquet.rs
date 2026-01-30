use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;
use parquet::basic::{Compression, ZstdLevel};
use parquet::column::reader::get_typed_column_reader;
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;

const ZSTD: &str = "zstd";

#[derive(Clone, Debug, Parser)]
pub struct ParquetConfig {
  #[arg(long, value_parser=str_to_compression, default_value = "uncompressed")]
  compression: Compression,
  // Larger group sizes work better on some datasets, and smaller ones on
  // others, sometimes with dramatic impact.
  // Based on experiments with zstd compression, 2^20 seems like a good default.
  #[arg(long, default_value = "1048576")]
  group_size: usize,
  #[arg(long, value_parser=str_to_encoding, default_value = "dictionary")]
  int_encoding: Encoding,
  #[arg(long, value_parser=str_to_encoding, default_value = "dictionary")]
  float_encoding: Encoding,
}

#[derive(Clone, Debug)]
struct Encoding {
  dictionary: bool,
  basic: Option<parquet::basic::Encoding>,
}

fn str_to_compression(s: &str) -> Result<Compression> {
  let res = match s.to_lowercase().as_str() {
    "uncompressed" => Compression::UNCOMPRESSED,
    "snappy" => Compression::SNAPPY,
    _ => {
      if let Some(zstd_level_str) = s.strip_prefix(ZSTD) {
        let level = if zstd_level_str.is_empty() {
          ZstdLevel::default()
        } else {
          ZstdLevel::try_new(zstd_level_str.parse::<i32>()?)?
        };
        Compression::ZSTD(level)
      } else {
        return Err(anyhow!("unknown parquet codec {}", s));
      }
    }
  };
  Ok(res)
}

fn compression_to_string(compression: &Compression) -> String {
  match compression {
    Compression::UNCOMPRESSED => "uncompressed".to_string(),
    Compression::SNAPPY => "snappy".to_string(),
    Compression::ZSTD(level) => format!("{}{}", ZSTD, level.compression_level()),
    _ => unreachable!(),
  }
}

fn str_to_encoding(s: &str) -> Result<Encoding> {
  let res = match s.to_lowercase().as_str() {
    "plain" => Encoding {
      dictionary: false,
      basic: Some(parquet::basic::Encoding::PLAIN),
    },
    "delta" => Encoding {
      dictionary: false,
      basic: Some(parquet::basic::Encoding::DELTA_BINARY_PACKED),
    },
    "byte_stream_split" => Encoding {
      dictionary: false,
      basic: Some(parquet::basic::Encoding::BYTE_STREAM_SPLIT),
    },
    "dictionary" => Encoding {
      dictionary: true,
      basic: None,
    },
    _ => return Err(anyhow!("unknown parquet encoding {}", s)),
  };
  Ok(res)
}

fn encoding_to_string(encoding: &Encoding) -> String {
  if encoding.dictionary {
    return "dictionary".to_string();
  }

  let s = match encoding.basic {
    Some(parquet::basic::Encoding::PLAIN) => "plain",
    Some(parquet::basic::Encoding::DELTA_BINARY_PACKED) => "delta",
    Some(parquet::basic::Encoding::BYTE_STREAM_SPLIT) => "byte_stream_split",
    _ => unreachable!(),
  };
  s.to_string()
}

// This approach compresses the vector as
impl CodecInternal for ParquetConfig {
  fn name(&self) -> &'static str {
    "parquet"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![
      (
        "compression",
        compression_to_string(&self.compression),
      ),
      ("group-size", self.group_size.to_string()),
      (
        "int-encoding",
        encoding_to_string(&self.int_encoding),
      ),
      (
        "float-encoding",
        encoding_to_string(&self.float_encoding),
      ),
    ]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let mut res = Vec::new();
    let message_type = format!(
      "message schema {{ REQUIRED {} nums; }}",
      T::PARQUET_DTYPE_STR
    );
    let schema = Arc::new(parse_message_type(&message_type).unwrap());

    let encoding = if T::PARQUET_DTYPE_STR.contains("INT") {
      &self.int_encoding
    } else {
      &self.float_encoding
    };

    let mut properties_builder = WriterProperties::builder()
      .set_writer_version(WriterVersion::PARQUET_2_0)
      .set_compression(self.compression)
      .set_dictionary_enabled(encoding.dictionary);
    if let Some(basic) = encoding.basic {
      properties_builder = properties_builder.set_encoding(basic);
    }

    let mut writer = SerializedFileWriter::new(
      &mut res,
      schema,
      Arc::new(properties_builder.build()),
    )
    .unwrap();

    for col_chunk in nums.chunks(self.group_size) {
      let mut row_group_writer = writer.next_row_group().unwrap();
      let mut col_writer = row_group_writer.next_column().unwrap().unwrap();
      let typed = col_writer.typed::<T::Parquet>();
      let parquet_vec;
      let parquet_slice = if T::TRANSMUTABLE {
        T::transmute_nums_to_parquet(col_chunk)
      } else {
        parquet_vec = T::copy_nums_to_parquet(col_chunk);
        &parquet_vec
      };
      typed.write_batch(parquet_slice, None, None).unwrap();
      col_writer.close().unwrap();
      row_group_writer.close().unwrap();
    }
    writer.close().unwrap();

    res
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    // couldn't find a way to make a parquet reader without a fully copy of the compressed bytes;
    // maybe this can be improved
    let reader = SerializedFileReader::new(bytes::Bytes::from(bytes.to_vec())).unwrap();

    let parquet_meta = reader.metadata();
    let mut n = 0;
    for row_group_meta in parquet_meta.row_groups() {
      n += row_group_meta.num_rows();
    }

    let mut res = Vec::with_capacity(n as usize);
    for i in 0..parquet_meta.num_row_groups() {
      let row_group_reader = reader.get_row_group(i).unwrap();
      let mut col_reader =
        get_typed_column_reader::<T::Parquet>(row_group_reader.get_column_reader(0).unwrap());
      col_reader
        .read_records(usize::MAX, None, None, &mut res)
        .unwrap();
    }

    T::parquet_to_nums(res)
  }
}
