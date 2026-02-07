use std::ops::AddAssign;
use std::time::{Duration, SystemTime};

use rand::Rng;

use q_compress::data_types::{NumberLike, TimestampMicros96};
use q_compress::errors::QCompressResult;
use q_compress::wrapped::ChunkSpec;
use q_compress::{wrapped, Compressor, CompressorConfig};

const BASE_DIR: &str = "q_compress/assets";

fn encode_usize(x: usize) -> [u8; 4] {
  (x as u32).to_be_bytes()
}

pub fn wrapped_compress<T: NumberLike>(
  nums: &[T],
  config: CompressorConfig,
  sizess: Vec<Vec<usize>>,
) -> QCompressResult<Vec<u8>> {
  let mut res = Vec::new();

  let mut compressor = wrapped::Compressor::<T>::from_config(config);
  compressor.header()?;
  let header = compressor.drain_bytes();
  res.extend(encode_usize(header.len()));
  res.extend(encode_usize(sizess.len()));
  res.extend(header);

  let mut start = 0;
  for sizes in sizess {
    let end = start + sizes.iter().sum::<usize>();
    let chunk_nums = &nums[start..end];
    start = end;
    let spec = ChunkSpec::default().with_page_sizes(sizes.clone());

    compressor.chunk_metadata(chunk_nums, &spec)?;
    let meta = compressor.drain_bytes();
    res.extend(encode_usize(meta.len()));
    res.extend(encode_usize(sizes.len()));
    res.extend(meta);

    for size in sizes {
      compressor.data_page()?;
      let page = compressor.drain_bytes();
      res.extend(encode_usize(page.len()));
      res.extend(encode_usize(size));
      res.extend(page);
    }
  }

  Ok(res)
}

#[derive(Clone, Copy, Debug)]
enum Mode {
  Standalone,
  Wrapped,
}

fn get_bytes<T: NumberLike>(nums: &[T], config: CompressorConfig, mode: Mode) -> Vec<u8> {
  match mode {
    Mode::Standalone => {
      let mut compressor = Compressor::<T>::from_config(config);
      compressor.simple_compress(nums)
    }
    Mode::Wrapped => {
      let n = nums.len();
      let sizess = vec![vec![1], vec![(n - 1) / 2, n / 2]];
      wrapped_compress(&nums, config, sizess).unwrap()
    }
  }
}

fn write_case<T: NumberLike>(
  version: &str,
  case_version: &str,
  name: &str,
  nums: Vec<T>,
  config: CompressorConfig,
  mode: Mode,
) {
  if version != case_version {
    return;
  }

  let compressed = get_bytes(&nums, config, mode);

  let raw = nums
    .iter()
    // .flat_map(|&x| T::bytes_from(x)) // for 0.4 to 0.5
    .flat_map(|&x| x.to_bytes()) // for 0.6+
    .collect::<Vec<u8>>();
  std::fs::write(
    format!("{}/v{}_{}.qco", BASE_DIR, version, name),
    compressed,
  )
  .expect("write qco");
  std::fs::write(
    format!("{}/v{}_{}.bin", BASE_DIR, version, name),
    raw,
  )
  .expect("write bin");
}

fn main() {
  let version = env!("CARGO_PKG_VERSION").to_string();
  let mut rng = rand::thread_rng();

  write_case(
    &version,
    "0.4.0",
    "i64_empty",
    Vec::<i64>::new(),
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut i32_2k = Vec::new();
  for _ in 0..2000 {
    let num = if rng.gen_bool(0.25) {
      rng.gen_range(0_i32..100)
    } else {
      rng.gen_range(1000_i32..1100)
    };
    i32_2k.push(num);
  }
  write_case(
    &version,
    "0.4.0",
    "i32_2k",
    i32_2k,
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut f32_2k = Vec::new();
  for _ in 0..2000 {
    f32_2k.push(10.0 / (1.0 + 9.0 * rng.gen::<f32>()));
  }
  write_case(
    &version,
    "0.4.0",
    "f32_2k",
    f32_2k,
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut bool_sparse_2k = Vec::new();
  for _ in 0..2000 {
    bool_sparse_2k.push(rng.gen_bool(0.1));
  }
  write_case(
    &version,
    "0.4.0",
    "bool_sparse_2k",
    bool_sparse_2k,
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut timestamp_deltas_2k = Vec::new();
  let mut t = SystemTime::now();
  for _ in 0..2000 {
    t.add_assign(Duration::from_secs_f64(
      0.5 + rng.gen::<f64>(),
    ));
    timestamp_deltas_2k.push(TimestampMicros96::from(t));
  }
  write_case(
    &version,
    "0.6.0",
    "timestamp_deltas_2k",
    timestamp_deltas_2k,
    CompressorConfig::default().with_delta_encoding_order(1),
    Mode::Standalone,
  );

  let mut dispersed_shorts = Vec::new();
  for _ in 0..64 {
    for i in 0_u16..20 {
      dispersed_shorts.push(i * 4);
    }
  }
  // really using 0.9.1
  write_case(
    &version,
    "0.9.1",
    "dispersed_shorts",
    dispersed_shorts,
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut varied_gcds = Vec::new();
  let mut same_gcds = Vec::new();
  for i in 0..(1 << 11) {
    varied_gcds.push(i as f32);
    same_gcds.push(i * 111);
  }
  write_case(
    &version,
    "0.10.0",
    "varied_gcds",
    varied_gcds,
    CompressorConfig::default(),
    Mode::Standalone,
  );
  write_case(
    &version,
    "0.10.0",
    "same_gcds",
    same_gcds,
    CompressorConfig::default(),
    Mode::Standalone,
  );

  let mut wrapped_brownian = Vec::new();
  let mut s = 100.0;
  for _ in 0..1000 {
    wrapped_brownian.push(s);
    s += rng.gen::<f32>();
  }
  write_case(
    &version,
    "0.11.2",
    "wrapped_brownian",
    wrapped_brownian,
    CompressorConfig::default().with_delta_encoding_order(1),
    Mode::Wrapped,
  );
}
