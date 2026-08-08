mod decoder;
mod vocab;

use std::path::Path;
use std::time::Instant;

use ndarray::Ix3;
use ort::session::Session;
use ort::value::Tensor;

use self::decoder::decode_autoregressive;
use self::vocab::Vocab;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

/// Known Canary model variants, auto-detected from vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanaryVariant {
    /// Canary Flash models (180M Flash, 1B Flash) — 4 languages.
    Flash,
    /// Canary 1B v2 — 27 languages.
    V2,
}

const FLASH_LANGUAGES: &[&str] = &["en", "de", "es", "fr"];

const V2_LANGUAGES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];

impl CanaryVariant {
    fn detect(vocab_size: usize) -> Self {
        if vocab_size < 10_000 {
            CanaryVariant::Flash
        } else {
            CanaryVariant::V2
        }
    }

    fn name(self) -> &'static str {
        match self {
            CanaryVariant::Flash => "Canary Flash",
            CanaryVariant::V2 => "Canary 1B v2",
        }
    }

    fn languages(self) -> &'static [&'static str] {
        match self {
            CanaryVariant::Flash => FLASH_LANGUAGES,
            CanaryVariant::V2 => V2_LANGUAGES,
        }
    }
}

/// Per-model inference parameters for Canary.
#[derive(Debug, Clone)]
pub struct CanaryParams {
    /// Source language hint (e.g. "en", "de"). Defaults to "en".
    pub language: Option<String>,
    /// Target language for translation (e.g. "en"). Defaults to source language.
    pub target_language: Option<String>,
    /// Punctuation and capitalization. When true, the model adds proper punctuation
    /// and capitalization to the output. When false, output is more literal/raw.
    /// Defaults to true.
    pub use_pnc: bool,
    /// Inverse text normalization. When true, spoken numbers and quantities are
    /// converted to written form (e.g. "one hundred twenty three" → "123").
    /// Only supported on V2 models; silently ignored on Flash models.
    /// Defaults to true.
    pub use_itn: bool,
    /// Maximum number of tokens to generate. Defaults to 1024.
    pub max_sequence_length: usize,
}

impl Default for CanaryParams {
    fn default() -> Self {
        Self {
            language: None,
            target_language: None,
            use_pnc: true,
            use_itn: true,
            max_sequence_length: 1024,
        }
    }
}

/// Canary speech model backed by three ONNX sessions (preprocessor, encoder, decoder).
pub struct CanaryModel {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    vocab: Vocab,
    variant: CanaryVariant,
    encoder_format: CanaryEncoderFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanaryEncoderFormat {
    FluidInference,
    Sherpa,
}

impl CanaryModel {
    /// Load a Canary model from `model_dir`.
    ///
    /// Expected directory contents:
    /// - `nemo128.onnx` (preprocessor, always FP32)
    /// - `encoder-model[.int8|.fp16].onnx` (quantization-aware)
    /// - `decoder-model[.int8|.fp16].onnx` (quantization-aware)
    /// - `vocab.txt`
    pub fn load(
        model_dir: &Path,
        quantization: &super::Quantization,
    ) -> Result<Self, TranscribeError> {
        if !model_dir.exists() {
            return Err(TranscribeError::ModelNotFound(model_dir.to_path_buf()));
        }

        let load_start = Instant::now();

        // Preprocessor is always FP32
        let preprocessor_path = model_dir.join("nemo128.onnx");
        log::info!(
            "Loading Canary preprocessor from {:?}...",
            preprocessor_path
        );
        let preprocessor = super::session::create_session(&preprocessor_path)?;

        // Encoder and decoder respect quantization
        let encoder_path =
            resolve_canary_model_path(model_dir, &["encoder-model", "encoder"], quantization);
        log::info!("Loading Canary encoder from {:?}...", encoder_path);
        let encoder = super::session::create_session(&encoder_path)?;
        let encoder_format = detect_encoder_format(&encoder)?;

        let decoder_path =
            resolve_canary_model_path(model_dir, &["decoder-model", "decoder"], quantization);
        log::info!("Loading Canary decoder from {:?}...", decoder_path);
        let decoder = super::session::create_session(&decoder_path)?;

        // Vocabulary
        let vocab_path = crate::decode::tokens::resolve_vocab_path(model_dir);
        let vocab = Vocab::load(&vocab_path)?;

        let variant = CanaryVariant::detect(vocab.size());
        log::info!(
            "Canary model loaded in {:.2?} (variant: {:?}, vocab: {} tokens)",
            load_start.elapsed(),
            variant,
            vocab.size()
        );

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            vocab,
            variant,
            encoder_format,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &CanaryParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let src_lang = params.language.as_deref().unwrap_or("en");
        let tgt_lang = params.target_language.as_deref().unwrap_or(src_lang);

        // Flash models don't support ITN — silently disable to avoid empty output
        let use_itn = params.use_itn && self.variant != CanaryVariant::Flash;

        let total_start = Instant::now();

        // --- Step 1: Preprocess audio -> mel features ---
        let preprocess_start = Instant::now();
        let num_samples = samples.len();

        log::debug!("Preprocessor input: waveforms shape [1, {}]", num_samples);

        let waveforms = Tensor::from_array((
            vec![1i64, num_samples as i64],
            samples.to_vec().into_boxed_slice(),
        ))?;
        let waveforms_lens =
            Tensor::from_array((vec![1i64], vec![num_samples as i64].into_boxed_slice()))?;

        let mut preprocess_out = self.preprocessor.run(ort::inputs![
            "waveforms" => waveforms,
            "waveforms_lens" => waveforms_lens
        ])?;

        log::debug!(
            "Preprocessor output: features shape {:?} ({:.2?})",
            preprocess_out["features"].shape(),
            preprocess_start.elapsed()
        );

        let features = preprocess_out
            .remove("features")
            .ok_or_else(|| TranscribeError::Inference("Missing features output".to_string()))?;
        let features_lens = preprocess_out.remove("features_lens").ok_or_else(|| {
            TranscribeError::Inference("Missing features_lens output".to_string())
        })?;

        // --- Step 2: Encode mel features -> encoder embeddings ---
        let encode_start = Instant::now();

        let mut encoder_out = match self.encoder_format {
            CanaryEncoderFormat::FluidInference => self.encoder.run(ort::inputs![
                "audio_signal" => features,
                "length" => features_lens
            ])?,
            CanaryEncoderFormat::Sherpa => {
                let features = features
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<Ix3>()
                    .map_err(|e| TranscribeError::Inference(e.to_string()))?
                    .permuted_axes([0, 2, 1])
                    .to_owned()
                    .into_dyn();
                let features_lens = features_lens.try_extract_array::<i64>()?.to_owned();
                let features = Tensor::from_array(features)?;
                let features_lens = Tensor::from_array(features_lens)?;

                self.encoder.run(ort::inputs![
                    "x" => features,
                    "x_len" => features_lens
                ])?
            }
        };

        log::debug!(
            "Encoder output: embeddings shape {:?}, mask shape {:?} ({:.2?})",
            encoder_out[encoder_embeddings_output_name(self.encoder_format)].shape(),
            encoder_out[encoder_mask_output_name(self.encoder_format)].shape(),
            encode_start.elapsed()
        );

        // Pass outputs directly to decoder (no data copy)
        let encoder_embeddings = encoder_out
            .remove(encoder_embeddings_output_name(self.encoder_format))
            .ok_or_else(|| {
                TranscribeError::Inference("Missing encoder embeddings output".to_string())
            })?;
        let encoder_mask = encoder_out
            .remove(encoder_mask_output_name(self.encoder_format))
            .ok_or_else(|| TranscribeError::Inference("Missing encoder mask output".to_string()))?;

        // --- Step 3: Build prompt tokens ---
        let prompt_tokens = self
            .vocab
            .build_prompt(src_lang, tgt_lang, params.use_pnc, use_itn)?;

        log::debug!(
            "Prompt tokens ({}): {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );

        // --- Step 4: Autoregressive decoding ---
        let decode_start = Instant::now();

        let text = decode_autoregressive(
            &mut self.decoder,
            &encoder_embeddings,
            &encoder_mask,
            prompt_tokens,
            &self.vocab,
            params.max_sequence_length,
        )?;

        log::debug!("Decoding completed in {:.2?}", decode_start.elapsed());
        log::info!(
            "Transcription completed in {:.2?}: \"{}\"",
            total_start.elapsed(),
            text
        );

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}

fn resolve_canary_model_path(
    dir: &Path,
    names: &[&str],
    quantization: &super::Quantization,
) -> std::path::PathBuf {
    for name in names {
        let path = super::session::resolve_model_path(dir, name, quantization);
        if path.exists() {
            return path;
        }
    }

    super::session::resolve_model_path(dir, names[0], quantization)
}

fn detect_encoder_format(encoder: &Session) -> Result<CanaryEncoderFormat, TranscribeError> {
    if encoder
        .inputs()
        .iter()
        .any(|input| input.name() == "audio_signal")
    {
        return Ok(CanaryEncoderFormat::FluidInference);
    }

    if encoder.inputs().iter().any(|input| input.name() == "x") {
        return Ok(CanaryEncoderFormat::Sherpa);
    }

    Err(TranscribeError::Inference(
        "Unsupported Canary encoder inputs: expected audio_signal or x".to_string(),
    ))
}

fn encoder_embeddings_output_name(format: CanaryEncoderFormat) -> &'static str {
    match format {
        CanaryEncoderFormat::FluidInference => "encoder_embeddings",
        CanaryEncoderFormat::Sherpa => "enc_states",
    }
}

fn encoder_mask_output_name(format: CanaryEncoderFormat) -> &'static str {
    match format {
        CanaryEncoderFormat::FluidInference => "encoder_mask",
        CanaryEncoderFormat::Sherpa => "enc_mask",
    }
}

impl SpeechModel for CanaryModel {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            name: self.variant.name(),
            engine_id: "canary",
            sample_rate: 16000,
            languages: self.variant.languages(),
            supports_timestamps: false,
            supports_translation: true,
            supports_streaming: false,
        }
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let src_lang = options.language.as_deref().unwrap_or("en");
        let tgt_lang = if options.translate { "en" } else { src_lang };
        let params = CanaryParams {
            language: Some(src_lang.to_string()),
            target_language: Some(tgt_lang.to_string()),
            ..Default::default()
        };
        self.transcribe_with(samples, &params)
    }
}
