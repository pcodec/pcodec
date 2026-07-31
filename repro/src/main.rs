use std::panic;

// (1) API side: ModeSpec::TryFloatQuant(k) is never validated against the
// float's latent width, so a k >= BITS shifts out of range in split_latents.
fn api_side(k: u32) {
  let cfg = pco::ChunkConfig::default().with_mode_spec(pco::ModeSpec::TryFloatQuant(k));
  let nums = vec![1.0f64, 2.0, 3.0];
  print!("compress f64 with TryFloatQuant({:<3}) -> ", k);
  match panic::catch_unwind(|| pco::standalone::simple_compress(&nums, &cfg)) {
    Ok(Ok(v)) => println!("Ok({} bytes)", v.len()),
    Ok(Err(e)) => println!("Err({})", e),
    Err(_) => println!("*** PANIC ***"),
  }
}

// (2) Wire side: the mode payload stores k in 8 bits with no upper bound
// check on read, so a patched byte reaches join_latents with k up to 255.
fn wire_side() {
  let cfg = pco::ChunkConfig::default().with_mode_spec(pco::ModeSpec::TryFloatQuant(20));
  let nums: Vec<f64> = (0..300).map(|i| (i as f64) * 0.25).collect();
  let good = pco::standalone::simple_compress(&nums, &cfg).unwrap();
  println!("valid FloatQuant file: {} bytes", good.len());
  // Brute-force a single-byte patch that raises k past 64.
  // skip the standalone header (bytes 0..14): patching n_hint there triggers a
  // different, unrelated bug (unbounded Vec::with_capacity).
  for idx in 14..good.len() {
    for val in 0u16..=255 {
      let mut bad = good.clone();
      bad[idx] = val as u8;
      let r = panic::catch_unwind(|| pco::standalone::simple_decompress::<f64>(&bad));
      if r.is_err() {
        println!("  single-byte patch at offset {} = {:#04x} -> *** PANIC on decompress ***", idx, val);
        return;
      }
    }
  }
  println!("  no single-byte patch panicked");
}

fn main() {
  if std::env::args().nth(1).as_deref() == Some("show") {
    let cfg = pco::ChunkConfig::default().with_mode_spec(pco::ModeSpec::TryFloatQuant(20));
    let nums: Vec<f64> = (0..300).map(|i| (i as f64) * 0.25).collect();
    let mut bad = pco::standalone::simple_compress(&nums, &cfg).unwrap();
    bad[16] = 0x59;
    let _ = pco::standalone::simple_decompress::<f64>(&bad);
    return;
  }
  panic::set_hook(Box::new(|_| {}));
  for k in [10u32, 52, 63, 64, 100, 255] {
    api_side(k);
  }
  wire_side();
}
