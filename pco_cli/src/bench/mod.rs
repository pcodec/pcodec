#![allow(clippy::uninit_vec)]

use std::collections::HashMap;
use std::convert::TryInto;
use std::fs;
use std::ops::AddAssign;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{anyhow, Result};
use arrow::datatypes::{DataType, Schema};
use clap::{Args, Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use tabled::settings::object::Columns;
use tabled::settings::{Alignment, Modify, Style};
use tabled::{Table, Tabled};

use pco::data_types::NumberType;
use pco::match_number_enum;

use crate::bench::codecs::CodecConfig;
use crate::input::{Format, InputColumnOpt, InputFileOpt};
use crate::{arrow_handlers, dtypes, input, parse, utils};

mod codecs;
pub mod handler;

const DEFAULT_BINARY_DIR: &str = "data/binary";
// if this delta order is specified, use a dataset-specific order

#[derive(Clone, Debug, ValueEnum)]
pub enum Units {
  Linear,
  Inverse,
  All,
}

/// Run benchmarks on datasets originating from another format.
///
/// This supports various input formats, various codecs (add even more with the
/// full_bench cargo feature), and configurations for each codec.
///
/// The input format does not affect performance; all input numbers are
/// loaded into memory prior to benchmarking each dataset.
/// By default, if no inputs are specified, the bench will use the
/// relative directory `data/binary/` as binary input.
#[derive(Clone, Debug, Parser)]
pub struct BenchOpt {
  /// Comma-separated list of codecs to benchmark, optionally with
  /// colon-separated configurations.
  ///
  /// For example, setting this to
  /// `zstd,zstd:level=10,pco:level=9:delta=Consecutive@1`
  /// will compare 3 codecs: zstd at default compression level (3), zstd at
  /// level 10, and pco at level 9 with 1st order delta encoding.
  ///
  /// To see what valid configurations look like, try entering an invalid one.
  #[arg(long, short, default_value = "pco", value_parser = CodecConfig::from_str, value_delimiter = ',')]
  pub codecs: Vec<CodecConfig>,
  /// Comma-separated substrings of datasets or column names to benchmark.
  /// By default all datasets are run.
  #[arg(long, short, default_values_t = Vec::<String>::new(), value_delimiter = ',')]
  pub datasets: Vec<String>,
  /// Filter down to datasets or columns matching this Arrow data type,
  /// e.g. i32 or micros.
  #[arg(long, default_values_t = Vec::<DataType>::new(), value_parser = parse::arrow_dtype, value_delimiter = ',')]
  pub dtypes: Vec<DataType>,
  /// Number of iterations to run each codec x dataset combination for
  /// better estimation of durations.
  /// The median duration is kept.
  #[arg(long, default_value = "10")]
  pub iters: usize,
  /// How many redundant, parallel threads to use for each iteration; i.e. the
  /// true number of iterations will be `codecs * datasets * iters * threads`.
  /// Only available with the full_bench feature.
  /// This is useful for measuring multithreaded performance, which is
  /// generally worse due to sharing cache and RAM bandwidth.
  #[arg(long)]
  pub threads: Option<usize>,
  /// How many numbers to limit each dataset to.
  #[arg(long, short)]
  pub limit: Option<usize>,
  /// CSV to write the aggregate results of this command to.
  /// Overwrites any rows with the same input name and codec config.
  /// Columns of output CSV:
  /// input_name, codec, compression_time/s, decompress_time/s, compressed_size/bytes
  #[arg(long)]
  pub results_csv: Option<PathBuf>,
  /// What units to print the results table in.
  /// Linear units are (de)compression time and size, whereas inverse units are
  /// (de)compression speed and ratio.
  /// Does not affect the results CSV.
  #[arg(short, long, default_value = "linear")]
  pub units: Units,
  /// Name of the input data to use in the --results-csv output.
  /// If you're not writing the results to a CSV, ignore this.
  #[arg(long)]
  pub input_name: Option<String>,
  #[command(flatten)]
  pub input: InputFileOpt,
  #[command(flatten)]
  pub iter_opt: IterOpt,
}

#[derive(Clone, Debug, Args)]
pub struct IterOpt {
  #[arg(long)]
  pub no_compress: bool,
  #[arg(long)]
  pub no_decompress: bool,
  /// Skip assertions that all the numbers came back bitwise identical.
  ///
  /// This does not affect benchmark timing.
  #[arg(long)]
  pub no_assertions: bool,
  /// Optionally, a directory to save the compressed data to.
  /// Will overwrite conflicting files.
  #[arg(long)]
  pub save_dir: Option<PathBuf>,
}

impl BenchOpt {
  pub fn includes_dataset(&self, dtype: &DataType, name: &str) -> bool {
    if dtypes::from_arrow(dtype).is_err()
      || (!self.dtypes.is_empty() && !self.dtypes.contains(dtype))
    {
      return false;
    }

    self.datasets.is_empty()
      || self
        .datasets
        .iter()
        .any(|allowed_substr| name.contains(allowed_substr))
  }
}

pub struct Precomputed {
  compressed: Vec<u8>,
}

fn make_progress_bar(n_columns: usize, opt: &BenchOpt) -> ProgressBar {
  ProgressBar::new(
    (opt.codecs.len() * n_columns * opt.threads.unwrap_or(1) * (opt.iters + 1)) as u64,
  )
  .with_message("iters")
  .with_style(
    ProgressStyle::with_template("[{elapsed_precise}] {wide_bar} {pos}/{len} {msg} ").unwrap(),
  )
}

fn median_duration(mut durations: Vec<Duration>) -> Duration {
  durations.sort_unstable();
  let lo = durations[(durations.len() - 1) / 2];
  let hi = durations[durations.len() / 2];
  (lo + hi) / 2
}

fn display_duration(duration: &Duration) -> String {
  format!("{:?}", duration)
}

#[derive(Clone, Default, Tabled)]
pub struct BenchStat {
  #[tabled(display = "display_duration")]
  pub compress_dt: Duration,
  #[tabled(display = "display_duration")]
  pub decompress_dt: Duration,
  pub compressed_size: usize,
  #[tabled(skip)]
  pub uncompressed_size: usize,
}

#[derive(Clone, Default, Tabled)]
pub struct InvStat {
  pub compress_mb_per_s: f32,
  pub decompress_mb_per_s: f32,
  pub compression_ratio: f32,
}

impl AddAssign for BenchStat {
  fn add_assign(&mut self, rhs: Self) {
    self.compressed_size += rhs.compressed_size;
    self.compress_dt += rhs.compress_dt;
    self.decompress_dt += rhs.decompress_dt;
    self.uncompressed_size += rhs.uncompressed_size
  }
}

impl BenchStat {
  fn aggregate_median(benches: &[BenchStat]) -> Self {
    let BenchStat {
      compressed_size,
      uncompressed_size,
      ..
    } = benches[0];
    let compress_dts = benches
      .iter()
      .map(|bench| bench.compress_dt)
      .collect::<Vec<_>>();
    let decompress_dts = benches
      .iter()
      .map(|bench| bench.decompress_dt)
      .collect::<Vec<_>>();

    BenchStat {
      compress_dt: median_duration(compress_dts),
      decompress_dt: median_duration(decompress_dts),
      compressed_size,
      uncompressed_size,
    }
  }
}

#[derive(Clone, Tabled)]
pub struct PrintStat {
  pub dataset: String,
  pub codec: String,
  #[tabled(inline)]
  pub bench_stat: BenchStat,
  #[tabled(inline)]
  pub inv_stat: InvStat,
}

impl PrintStat {
  pub fn new(dataset: String, codec: String, bench_stat: BenchStat) -> Self {
    let uncompressed_size = bench_stat.uncompressed_size as f32;
    let inv_stat = InvStat {
      compress_mb_per_s: uncompressed_size / (1_000_000.0 * bench_stat.compress_dt.as_secs_f32()),
      decompress_mb_per_s: uncompressed_size
        / (1_000_000.0 * bench_stat.decompress_dt.as_secs_f32()),
      compression_ratio: uncompressed_size / bench_stat.compressed_size as f32,
    };
    Self {
      dataset,
      codec,
      bench_stat,
      inv_stat,
    }
  }
}

fn core_dtype_to_str(dtype: NumberType) -> String {
  match_number_enum!(
    dtype,
    NumberType<T> => {
      utils::dtype_name::<T>()
    }
  )
}

fn handle_column(
  schema: &Schema,
  col_idx: usize,
  opt: &BenchOpt,
  progress_bar: ProgressBar,
) -> Result<Vec<PrintStat>> {
  let field = &schema.fields[col_idx];
  let reader = input::new_column_reader(schema, col_idx, &opt.input)?;
  let mut arrays = Vec::new();
  for array_result in reader {
    arrays.push(array_result?);
  }
  let handler = arrow_handlers::from_dtype(field.data_type())?;
  handler.bench(&arrays, field.name(), opt, progress_bar)
}

fn update_results_csv(
  aggregate_by_codec: &HashMap<String, BenchStat>,
  opt: &BenchOpt,
) -> Result<()> {
  // do nothing if the user didn't provide a results CSV
  let Some(results_csv) = opt.results_csv.as_ref() else {
    return Ok(());
  };

  let input_name = opt.input_name.as_ref().unwrap();

  let mut lines = if results_csv.exists() {
    // hacky split on commas, doesn't handle case when values contain weird characters
    let mut lines = HashMap::new();
    let contents = fs::read_to_string(results_csv)?;
    let mut is_header = true;
    for line in contents.split('\n') {
      if is_header {
        is_header = false;
        continue;
      }

      let fields: Vec<&str> = line.split(',').take(5).collect::<Vec<&str>>();
      if fields.len() == 1 && fields[0].is_empty() {
        // skip empty lines
        continue;
      }
      let fields: [&str; 5] = fields.clone().try_into().map_err(|_| {
        anyhow!(
          "existing results CSV row contained fewer than 5 fields: {:?}",
          fields
        )
      })?;
      let [dataset, codec, compress_dt, decompress_dt, compressed_size] = fields;
      let stat = BenchStat {
        compress_dt: Duration::from_secs_f32(compress_dt.parse()?),
        decompress_dt: Duration::from_secs_f32(decompress_dt.parse()?),
        compressed_size: compressed_size.parse()?,
        uncompressed_size: 0, // we don't know, and it doesn't matter
      };
      lines.insert(
        (dataset.to_string(), codec.to_string()),
        stat,
      );
    }
    lines
  } else {
    HashMap::new()
  };

  for (codec, stat) in aggregate_by_codec.iter() {
    let key = (input_name.to_string(), codec.to_string());
    let mut stat = stat.clone();
    if let Some(existing_stat) = lines.get(&key) {
      if opt.iter_opt.no_compress {
        stat.compress_dt = existing_stat.compress_dt;
      }
      if opt.iter_opt.no_decompress {
        stat.decompress_dt = existing_stat.decompress_dt;
      }
    }
    lines.insert(key, stat);
  }

  let mut output_lines = vec!["input,codec,compress_dt,decompress_dt,compressed_size".to_string()];
  let mut lines = lines.iter().collect::<Vec<_>>();
  lines.sort_unstable_by_key(|&(key, _)| key);
  for ((dataset, codec), stat) in lines {
    output_lines.push(format!(
      "{},{},{},{},{}",
      dataset,
      codec,
      stat.compress_dt.as_secs_f32(),
      stat.decompress_dt.as_secs_f32(),
      stat.compressed_size,
    ));
  }
  let output = output_lines.join("\n");
  fs::write(results_csv, output)?;

  Ok(())
}

fn print_stats(mut stats: Vec<PrintStat>, opt: &BenchOpt) -> Result<()> {
  if stats.is_empty() {
    return Err(anyhow!(
      "No datasets found that match filters"
    ));
  }

  let mut aggregate_by_codec: HashMap<String, BenchStat> = HashMap::new();
  for stat in &stats {
    aggregate_by_codec
      .entry(stat.codec.clone())
      .or_default()
      .add_assign(stat.bench_stat.clone());
  }
  stats.extend(opt.codecs.iter().map(|codec| {
    let codec = codec.to_string();
    let bench_stat = aggregate_by_codec.get(&codec).cloned().unwrap();
    PrintStat::new("<sum>".to_string(), codec, bench_stat)
  }));
  let mut table_builder = Table::builder(stats);
  match opt.units {
    Units::All => (),
    Units::Linear => {
      for _ in 0..3 {
        // Removing columns takes place immediately, so we remove the 5th one 3
        // times to delete columns 5, 6, 7.
        table_builder.remove_column(5);
      }
    }
    Units::Inverse => {
      for _ in 0..3 {
        table_builder.remove_column(2);
      }
    }
  }
  let table = table_builder
    .build()
    .with(Style::rounded())
    .with(Modify::new(Columns::new(2..)).with(Alignment::right()))
    .to_string();
  println!("{}", table);
  update_results_csv(&aggregate_by_codec, opt)
}

pub fn bench(mut opt: BenchOpt) -> Result<()> {
  if opt.results_csv.is_some() && opt.input_name.is_none() {
    return Err(anyhow!(
      "input-name must be specified when results-csv is"
    ));
  }
  if opt.threads.is_some() && !cfg!(feature = "full_bench") {
    return Err(anyhow!(
      "threads can only be specified when built with the full_bench feature"
    ));
  }

  let input = &mut opt.input;
  if input.input.is_none() && input.input_format.is_none() {
    input.input = Some(PathBuf::from(DEFAULT_BINARY_DIR));
    input.input_format = Some(Format::Binary);
  }

  let schema = input::get_schema(&InputColumnOpt::default(), input)?;

  let col_idxs = schema
    .fields
    .iter()
    .enumerate()
    .filter_map(|(i, field)| {
      if opt.includes_dataset(field.data_type(), field.name()) {
        Some(i)
      } else {
        None
      }
    })
    .collect::<Vec<_>>();

  #[cfg(feature = "full_bench")]
  if let Some(threads) = opt.threads {
    use rayon::ThreadPoolBuilder;
    ThreadPoolBuilder::new()
      .num_threads(threads)
      .build_global()?;
  }

  let progress_bar = make_progress_bar(col_idxs.len(), &opt);
  let mut stats = Vec::new();
  for col_idx in col_idxs {
    stats.extend(handle_column(
      &schema,
      col_idx,
      &opt,
      progress_bar.clone(),
    )?);
  }
  progress_bar.finish_and_clear();

  print_stats(stats, &opt)
}
