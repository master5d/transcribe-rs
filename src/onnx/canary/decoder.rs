use std::borrow::Cow;

use ndarray::{Array2, Array3, Array4};
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::ValueType;
use ort::value::{DynValue, Tensor};

use super::vocab::Vocab;
use crate::decode::GreedyDecoder;
use crate::TranscribeError;

pub fn decode_autoregressive(
    decoder: &mut Session,
    encoder_embeddings: &DynValue,
    encoder_mask: &DynValue,
    prompt_tokens: Vec<i64>,
    vocab: &Vocab,
    max_sequence_length: usize,
) -> Result<String, TranscribeError> {
    if decoder
        .inputs()
        .iter()
        .any(|input| input.name() == "decoder_mems")
    {
        return decode_fluidinference(
            decoder,
            encoder_embeddings,
            encoder_mask,
            prompt_tokens,
            vocab,
            max_sequence_length,
        );
    }

    if decoder
        .inputs()
        .iter()
        .any(|input| input.name() == "decoder_mems_list_0")
    {
        return decode_sherpa(
            decoder,
            encoder_embeddings,
            encoder_mask,
            prompt_tokens,
            vocab,
            max_sequence_length,
        );
    }

    Err(TranscribeError::Inference(
        "Unsupported Canary decoder inputs: expected decoder_mems or decoder_mems_list_0"
            .to_string(),
    ))
}

fn decode_fluidinference(
    decoder: &mut Session,
    encoder_embeddings: &DynValue,
    encoder_mask: &DynValue,
    prompt_tokens: Vec<i64>,
    vocab: &Vocab,
    max_sequence_length: usize,
) -> Result<String, TranscribeError> {
    let (num_layers, hidden_dim) = extract_decoder_mems_shape(decoder)?;

    log::debug!(
        "Decoder cache dimensions: num_layers={}, hidden_dim={}",
        num_layers,
        hidden_dim
    );

    let empty_cache = Array4::<f32>::zeros((num_layers, 1, 0, hidden_dim));
    let mut decoder_mems: DynValue = Tensor::from_array(empty_cache)?.into_dyn();

    let eos_id = vocab.eos_token_id();
    let mut greedy = GreedyDecoder::new(eos_id);
    let mut all_tokens = prompt_tokens;

    // Limit decode steps so total tokens (prompt + generated) stays within
    // the model's position embedding table (typically 1024).
    let max_steps = max_sequence_length.saturating_sub(all_tokens.len());

    log::debug!(
        "Starting autoregressive decode with {} prompt tokens, max_steps={}",
        all_tokens.len(),
        max_steps
    );

    for step in 0..max_steps {
        let input_ids_tensor = if step == 0 {
            let len = all_tokens.len();
            let shape = vec![1i64, len as i64];
            Tensor::from_array((shape, all_tokens.clone().into_boxed_slice()))?
        } else {
            let last = *all_tokens
                .last()
                .ok_or_else(|| TranscribeError::Inference("Token list is empty".to_string()))?;
            Tensor::from_array((vec![1i64, 1i64], vec![last].into_boxed_slice()))?
        };

        let mut outputs = decoder.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "encoder_embeddings" => encoder_embeddings,
            "encoder_mask" => encoder_mask,
            "decoder_mems" => decoder_mems
        ])?;

        // Extract logits in a scoped borrow, then release before remove()
        let last_logits = {
            let (logits_shape, logits_data) =
                outputs["logits"].try_extract_tensor::<f32>().map_err(|e| {
                    TranscribeError::Inference(format!("Failed to extract logits: {e}"))
                })?;

            let seq_len = logits_shape[1] as usize;
            let vocab_size = logits_shape[2] as usize;

            let last_step_offset = (seq_len - 1) * vocab_size;
            logits_data[last_step_offset..last_step_offset + vocab_size].to_vec()
        };

        let next_token = match greedy.next_token(&last_logits) {
            Some(t) => t,
            None => {
                log::debug!("Decode stopped at step {}", step);
                break;
            }
        };

        log::debug!("Step {}: predicted token ID {}", step, next_token);

        all_tokens.push(next_token);

        // Take the KV cache directly from outputs (Arc clone, no data copy)
        decoder_mems = outputs.remove("decoder_hidden_states").ok_or_else(|| {
            TranscribeError::Inference("Missing decoder_hidden_states output".to_string())
        })?;
    }

    let text = vocab.decode_tokens(&all_tokens);
    Ok(text)
}

fn decode_sherpa(
    decoder: &mut Session,
    encoder_embeddings: &DynValue,
    encoder_mask: &DynValue,
    prompt_tokens: Vec<i64>,
    vocab: &Vocab,
    max_sequence_length: usize,
) -> Result<String, TranscribeError> {
    let (num_layers, hidden_dim) = extract_sherpa_decoder_mems_shape(decoder)?;

    log::debug!(
        "Sherpa decoder cache dimensions: num_layers={}, hidden_dim={}",
        num_layers,
        hidden_dim
    );

    let mut decoder_mems: Vec<DynValue> = (0..num_layers)
        .map(|_| {
            let empty_cache = Array3::<f32>::zeros((1, 0, hidden_dim));
            Tensor::from_array(empty_cache).map(|v| v.into_dyn())
        })
        .collect::<Result<_, _>>()?;

    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let (next_logits, next_mems) = run_sherpa_decoder_step(
            decoder,
            token,
            pos as i32,
            decoder_mems,
            encoder_embeddings,
            encoder_mask,
            num_layers,
        )?;
        logits = next_logits;
        decoder_mems = next_mems;
    }

    let eos_id = vocab.eos_token_id();
    let mut greedy = GreedyDecoder::new(eos_id);
    let mut generated = Vec::new();
    let max_steps = max_sequence_length.saturating_sub(prompt_tokens.len());

    for pos in 1..=max_steps {
        let next_token = match greedy.next_token(&logits) {
            Some(t) => t,
            None => {
                log::debug!("Decode stopped after {} generated tokens", generated.len());
                break;
            }
        };

        generated.push(next_token);

        let (next_logits, next_mems) = run_sherpa_decoder_step(
            decoder,
            next_token,
            pos as i32,
            decoder_mems,
            encoder_embeddings,
            encoder_mask,
            num_layers,
        )?;
        logits = next_logits;
        decoder_mems = next_mems;
    }

    Ok(vocab.decode_tokens(&generated))
}

fn run_sherpa_decoder_step(
    decoder: &mut Session,
    token: i64,
    position: i32,
    decoder_mems: Vec<DynValue>,
    encoder_embeddings: &DynValue,
    encoder_mask: &DynValue,
    num_layers: usize,
) -> Result<(Vec<f32>, Vec<DynValue>), TranscribeError> {
    let input_ids = Array2::from_shape_vec((1, 2), vec![token as i32, position])?.into_dyn();

    let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
        (
            Cow::Borrowed("decoder_input_ids"),
            SessionInputValue::from(ort::value::Value::from_array(input_ids)?),
        ),
        (
            Cow::Borrowed("enc_states"),
            SessionInputValue::from(encoder_embeddings),
        ),
        (
            Cow::Borrowed("enc_mask"),
            SessionInputValue::from(encoder_mask),
        ),
    ];

    for (idx, mem) in decoder_mems.into_iter().enumerate() {
        inputs.push((
            Cow::Owned(format!("decoder_mems_list_{idx}")),
            SessionInputValue::from(mem),
        ));
    }

    let mut outputs = decoder.run(inputs)?;

    let logits = {
        let (logits_shape, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| TranscribeError::Inference(format!("Failed to extract logits: {e}")))?;

        let seq_len = logits_shape[1] as usize;
        let vocab_size = logits_shape[2] as usize;
        let last_step_offset = (seq_len - 1) * vocab_size;
        logits_data[last_step_offset..last_step_offset + vocab_size].to_vec()
    };

    let next_mems = (0..num_layers)
        .map(|idx| {
            outputs
                .remove(format!("next_decoder_mem_list_{idx}").as_str())
                .ok_or_else(|| {
                    TranscribeError::Inference(format!(
                        "Missing next_decoder_mem_list_{idx} output"
                    ))
                })
        })
        .collect::<Result<_, _>>()?;

    Ok((logits, next_mems))
}

fn extract_decoder_mems_shape(decoder: &Session) -> Result<(usize, usize), TranscribeError> {
    let mems_input = decoder
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == "decoder_mems")
        .ok_or_else(|| {
            TranscribeError::Inference("Decoder model missing 'decoder_mems' input".to_string())
        })?;

    match mems_input.dtype() {
        ValueType::Tensor { shape, .. } => {
            let dims: &[i64] = &shape;
            if dims.len() != 4 {
                return Err(TranscribeError::Inference(format!(
                    "Expected 4D decoder_mems, got {}D",
                    dims.len()
                )));
            }

            let num_layers = dims[0];
            let hidden_dim = dims[3];

            if num_layers <= 0 || hidden_dim <= 0 {
                return Err(TranscribeError::Inference(format!(
                    "decoder_mems has dynamic num_layers ({}) or hidden_dim ({}); expected fixed",
                    num_layers, hidden_dim
                )));
            }

            Ok((num_layers as usize, hidden_dim as usize))
        }
        other => Err(TranscribeError::Inference(format!(
            "decoder_mems input is not a tensor: {:?}",
            other
        ))),
    }
}

fn extract_sherpa_decoder_mems_shape(decoder: &Session) -> Result<(usize, usize), TranscribeError> {
    let mut num_layers = 0;
    while decoder
        .inputs()
        .iter()
        .any(|input| input.name() == format!("decoder_mems_list_{num_layers}"))
    {
        num_layers += 1;
    }

    if num_layers == 0 {
        return Err(TranscribeError::Inference(
            "Sherpa decoder model missing decoder_mems_list_* inputs".to_string(),
        ));
    }

    let mems_input = decoder
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == "decoder_mems_list_0")
        .ok_or_else(|| {
            TranscribeError::Inference(
                "Sherpa decoder model missing decoder_mems_list_0 input".to_string(),
            )
        })?;

    match mems_input.dtype() {
        ValueType::Tensor { shape, .. } => {
            let dims: &[i64] = &shape;
            if dims.len() != 3 {
                return Err(TranscribeError::Inference(format!(
                    "Expected 3D decoder_mems_list_0, got {}D",
                    dims.len()
                )));
            }

            let hidden_dim = dims[2];
            if hidden_dim <= 0 {
                return Err(TranscribeError::Inference(format!(
                    "decoder_mems_list_0 has dynamic hidden_dim ({hidden_dim}); expected fixed"
                )));
            }

            Ok((num_layers, hidden_dim as usize))
        }
        other => Err(TranscribeError::Inference(format!(
            "decoder_mems_list_0 input is not a tensor: {:?}",
            other
        ))),
    }
}
