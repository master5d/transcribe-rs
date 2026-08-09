use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams};
use transcribe_rs::onnx::Quantization;

const DEFAULT_MODEL_DIR: &str =
    r"C:\telo\Efforts\On\echo\src-tauri\resources\models\parakeet-tdt-0.6b-v3-int8";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let wav_path = PathBuf::from(
        args.next()
            .ok_or("usage: transcribe_cli <audio.wav> [model_dir]")?,
    );
    let model_dir = args
        .next()
        .or_else(|| std::env::var("TRANSCRIBE_MODEL_DIR").ok())
        .unwrap_or_else(|| DEFAULT_MODEL_DIR.to_string());
    let model_path = PathBuf::from(&model_dir);

    eprintln!("[transcribe_cli] model: {:?}", model_path);
    eprintln!("[transcribe_cli] audio: {:?}", wav_path);

    let t0 = Instant::now();
    let mut model = ParakeetModel::load(&model_path, &Quantization::Int8)?;
    eprintln!("[transcribe_cli] model loaded in {:.2?}", t0.elapsed());

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let t1 = Instant::now();
    let result = model.transcribe_with(&samples, &ParakeetParams::default())?;
    eprintln!("[transcribe_cli] transcribed in {:.2?}", t1.elapsed());

    // stdout: transcript only (trimmed), so callers can capture it cleanly.
    print!("{}", result.text.trim());
    Ok(())
}
