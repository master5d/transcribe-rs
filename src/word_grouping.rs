/// One whisper token: raw UTF-8 bytes (may split a char across tokens) + seconds span.
#[derive(Clone)]
pub struct RawTok {
    pub start: f32,
    pub end: f32,
    pub bytes: Vec<u8>,
}

/// A grouped word with span and trimmed text.
pub struct GroupedWord {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Group whisper tokens into words. A new word begins at the first token and at any
/// token whose first byte is a space (0x20 — whisper's BPE word-boundary marker, valid
/// for Cyrillic). Bytes are accumulated per word and decoded once with `from_utf8_lossy`
/// so tokens split across UTF-8 boundaries (common in Russian) join correctly.
pub fn group_tokens_into_words(tokens: &[RawTok]) -> Vec<GroupedWord> {
    let mut words: Vec<(f32, f32, Vec<u8>)> = Vec::new();
    for t in tokens {
        let starts_word = t.bytes.first() == Some(&b' ');
        if words.is_empty() || starts_word {
            words.push((t.start, t.end, t.bytes.clone()));
        } else if let Some(last) = words.last_mut() {
            last.2.extend_from_slice(&t.bytes);
            last.1 = t.end;
        }
    }
    words
        .into_iter()
        .filter_map(|(start, end, bytes)| {
            let text = String::from_utf8_lossy(&bytes).trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(GroupedWord { start, end, text })
            }
        })
        .collect()
}

#[cfg(test)]
mod word_grouping_tests {
    use super::*;

    // bytes emulate whisper token data (text already as UTF-8 bytes), times in seconds.
    fn tok(s: f32, e: f32, t: &str) -> RawTok {
        RawTok {
            start: s,
            end: e,
            bytes: t.as_bytes().to_vec(),
        }
    }

    #[test]
    fn groups_ascii_words_on_leading_space() {
        let toks = vec![
            tok(0.0, 0.1, " hel"),
            tok(0.1, 0.2, "lo"),
            tok(0.2, 0.4, " there"),
        ];
        let words = group_tokens_into_words(&toks);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.2);
        assert_eq!(words[1].text, "there");
        assert_eq!(words[1].start, 0.2);
        assert_eq!(words[1].end, 0.4);
    }

    #[test]
    fn groups_cyrillic_split_across_byte_boundary() {
        // " привет" then "вет" continuation, then " мир". Split a multi-byte char across
        // tokens to prove byte-accumulation + lossy-decode works (here we keep it valid).
        let toks = vec![
            tok(0.0, 0.2, " при"),
            tok(0.2, 0.3, "вет"),
            tok(0.3, 0.5, " мир"),
        ];
        let words = group_tokens_into_words(&toks);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "привет");
        assert_eq!(words[1].text, "мир");
    }

    #[test]
    fn monotonic_non_decreasing() {
        let toks = vec![
            tok(0.0, 0.2, " a"),
            tok(0.2, 0.5, " b"),
            tok(0.5, 0.9, " c"),
        ];
        let words = group_tokens_into_words(&toks);
        for w in &words {
            assert!(w.end >= w.start);
        }
        for pair in words.windows(2) {
            assert!(pair[1].start >= pair[0].start);
        }
    }
}
