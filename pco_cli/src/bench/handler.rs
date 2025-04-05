use anyhow::Result;
use arrow::array::{ArrayRef, AsArray};
use indicatif::ProgressBar;

use crate::arrow_handlers::ArrowHandlerImpl;
use crate::bench::codecs::CodecConfig;
use crate::bench::{core_dtype_to_str, BenchOpt, BenchStat, PrintStat};
use crate::dtypes::{ArrowNumber, PcoNumber};
use crate::num_vec::NumVec;

pub trait BenchHandler {
  fn bench(
    &self,
    arrays: &[ArrayRef],
    name: &str,
    opt: &BenchOpt,
    progress_bar: ProgressBar,
  ) -> Result<Vec<PrintStat>>;
}

fn handle_for_codec_thread(
  num_vec: &NumVec,
  name: &str,
  codec: &CodecConfig,
  opt: &BenchOpt,
  progress_bar: ProgressBar,
  thread_idx: usize,
) -> Result<PrintStat> {
  let dataset = format!(
    "{}_{}",
    core_dtype_to_str(num_vec.dtype()),
    name,
  );
  let precomputed = codec.warmup_iter(num_vec, &dataset, &opt.iter_opt, thread_idx)?;
  progress_bar.inc(1);

  let mut benches = Vec::with_capacity(opt.iters);
  for _ in 0..opt.iters {
    benches.push(codec.stats_iter(num_vec, &precomputed, &opt.iter_opt)?);
    progress_bar.inc(1);
  }
  Ok(PrintStat::new(
    dataset,
    codec.to_string(),
    BenchStat::aggregate_median(&benches),
  ))
}

impl<P: ArrowNumber> BenchHandler for ArrowHandlerImpl<P> {
  fn bench(
    &self,
    arrays: &[ArrayRef],
    name: &str,
    opt: &BenchOpt,
    progress_bar: ProgressBar,
  ) -> Result<Vec<PrintStat>> {
    let arrow_nums: Vec<P::Native> = arrays
      .iter()
      .flat_map(|arr| arr.as_primitive::<P>().values().iter().cloned())
      .collect::<Vec<_>>();

    let nums = P::native_vec_to_pco(arrow_nums);
    let num_vec = P::Pco::make_num_vec(nums);

    let mut stats = Vec::new();
    let limited_num_vec;
    let mut num_vec_ref = &num_vec;
    if let Some(limit) = opt.limit {
      if limit < num_vec.n() {
        limited_num_vec = num_vec.truncated(limit);
        num_vec_ref = &limited_num_vec;
      }
    }
    for codec in &opt.codecs {
      let progress_bar = progress_bar.clone(); // trivial copy of an Arc<>

      #[cfg(feature = "rayon")]
      if let Some(threads) = opt.threads {
        // launch the warmup + iters for each thread, then aggregate as median
        // of medians
        use rayon::prelude::*;
        let new_stats = (0..threads)
          .into_par_iter()
          .map(move |thread_idx| {
            handle_for_codec_thread(
              num_vec_ref,
              name,
              codec,
              opt,
              progress_bar.clone(),
              thread_idx,
            )
          })
          .collect::<Result<Vec<_>>>()?;

        let PrintStat { dataset, codec, .. } = new_stats[0].clone();
        let thread_benches = new_stats
          .iter()
          .map(|stat| stat.bench_stat.clone())
          .collect::<Vec<_>>();
        stats.push(PrintStat::new(
          dataset,
          codec,
          BenchStat::aggregate_median(&thread_benches),
        ));
        continue;
      }

      stats.push(handle_for_codec_thread(
        num_vec_ref,
        name,
        codec,
        opt,
        progress_bar,
        0,
      )?);
    }

    Ok(stats)
  }
}
