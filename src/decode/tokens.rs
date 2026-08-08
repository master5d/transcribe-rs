use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Путь к словарю модели: `vocab.txt` (экспорты FluidInference) либо
/// `tokens.txt` (бандлы sherpa-onnx) — одно и то же по смыслу, разные имена.
/// Если нет ни одного, возвращает `vocab.txt`, чтобы ошибка называла
/// каноническое имя, а не последнее проверенное.
pub fn resolve_vocab_path(model_dir: &Path) -> std::path::PathBuf {
    let vocab = model_dir.join("vocab.txt");
    if vocab.exists() {
        return vocab;
    }
    let tokens = model_dir.join("tokens.txt");
    if tokens.exists() {
        return tokens;
    }
    vocab
}

/// Load a vocabulary file where each line is `token id`.
///
/// Returns a Vec indexed by token ID, and the blank token index.
/// Replaces `▁` (U+2581) with space in token strings.
///
/// Два формата в дикой природе, и они различаются тем, как записан ПРОБЕЛ:
/// экспорты FluidInference кодируют его маркером `▁` (`▁word 5`), а бандлы
/// sherpa-onnx пишут литеральный пробел отдельным токеном — строка выглядит как
/// `"  0"` (пробел-токен, пробел-разделитель, id). Поэтому разбираем строку
/// СПРАВА: последнее поле — id, всё до него — токен как есть. Разбор слева
/// (`split(' ')[0]`) на sherpa-словаре давал пустую строку вместо пробела, и
/// текст склеивался: `почувствоватьчтовыустроеныудобно…`.
pub fn load_vocab(path: &Path) -> Result<(Vec<String>, Option<i32>), std::io::Error> {
    let content = fs::read_to_string(path)?;

    let mut max_id = 0;
    let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
    let mut blank_idx: Option<i32> = None;

    for line in content.lines() {
        if let Some((token, id_str)) = line.trim_end_matches(['\r', '\n']).rsplit_once(' ') {
            if let Ok(id) = id_str.parse::<usize>() {
                let token = token.to_string();
                if token == "<blk>" {
                    blank_idx = Some(id as i32);
                }
                tokens_with_ids.push((token, id));
                max_id = max_id.max(id);
            }
        }
    }

    let mut vocab = vec![String::new(); max_id + 1];
    for (token, id) in tokens_with_ids {
        vocab[id] = token.replace('\u{2581}', " ");
    }

    log::info!("Loaded {} vocab tokens from {:?}", vocab.len(), path);
    Ok((vocab, blank_idx))
}

/// Symbol table mapping token IDs to strings.
///
/// Supports two file formats:
/// - `symbol id` (split on last whitespace, used by SenseVoice)
/// - Optionally base64-encoded symbols (for FunASR Nano models)
pub struct SymbolTable {
    id_to_sym: HashMap<i64, String>,
}

impl SymbolTable {
    /// Load a symbol table from a file where each line is `symbol id`.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() == 2 {
                if let Ok(id) = parts[0].parse::<i64>() {
                    id_to_sym.insert(id, parts[1].to_string());
                }
            }
        }

        log::info!("Loaded {} tokens from {:?}", id_to_sym.len(), path);
        Ok(Self { id_to_sym })
    }

    /// Decode all symbols from base64 (for FunASR Nano models).
    #[cfg(feature = "onnx")]
    pub fn apply_base64_decode(&mut self) {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        for sym in self.id_to_sym.values_mut() {
            if let Ok(bytes) = STANDARD.decode(sym.as_bytes()) {
                if let Ok(decoded) = String::from_utf8(bytes) {
                    *sym = decoded;
                }
            }
        }
    }

    pub fn get(&self, id: i64) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    pub fn get_or_empty(&self, id: i64) -> &str {
        self.id_to_sym.get(&id).map(|s| s.as_str()).unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    /// Формат sherpa-onnx: пробел записан ЛИТЕРАЛОМ отдельным токеном, поэтому
    /// строка выглядит как "  0" (токен, разделитель, id). Разбор слева терял
    /// его и склеивал слова.
    #[test]
    fn sherpa_literal_space_token_survives() {
        let dir = std::env::temp_dir().join("trs_vocab_sherpa");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write(&dir, "tokens.txt", "  0\nа 1\nб 2\n<blk> 3\n");
        let (vocab, blank) = load_vocab(&p).unwrap();
        assert_eq!(vocab[0], " ", "пробел-токен обязан выжить, иначе текст склеится");
        assert_eq!(vocab[1], "а");
        assert_eq!(blank, Some(3));
    }

    /// Формат FluidInference: пробел закодирован маркером U+2581.
    #[test]
    fn fluidinference_underscore_marker_still_maps_to_space() {
        let dir = std::env::temp_dir().join("trs_vocab_fi");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write(&dir, "vocab.txt", "\u{2581}word 5\nsuffix 6\n<blk> 7\n");
        let (vocab, blank) = load_vocab(&p).unwrap();
        assert_eq!(vocab[5], " word");
        assert_eq!(vocab[6], "suffix");
        assert_eq!(blank, Some(7));
    }

    #[test]
    fn resolve_vocab_prefers_vocab_then_tokens_then_canonical_name() {
        let dir = std::env::temp_dir().join("trs_vocab_resolve");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // ничего нет -> каноническое имя (чтобы ошибка называла vocab.txt)
        assert!(resolve_vocab_path(&dir).ends_with("vocab.txt"));
        // только sherpa-имя -> берём его
        write(&dir, "tokens.txt", "a 0\n");
        assert!(resolve_vocab_path(&dir).ends_with("tokens.txt"));
        // есть оба -> vocab.txt главнее
        write(&dir, "vocab.txt", "a 0\n");
        assert!(resolve_vocab_path(&dir).ends_with("vocab.txt"));
    }
}
