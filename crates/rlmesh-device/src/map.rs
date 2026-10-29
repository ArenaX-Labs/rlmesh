//! A robot map's part paths, in authored order, without parsing the map:
//! its top-level keys. The host normalises the map at `DeclareParts` and
//! refuses one it can't; the board needs only the paths (their order, and
//! `DeclareParts.seal`), so it doesn't carry the normaliser and a JSON
//! parser in flash.

use alloc::string::String;
use alloc::vec::Vec;

use rlmesh_session_core::wire::PartPath;

/// The top-level keys of `json`, in order, each a valid part path, or
/// `None` when `json` isn't one JSON object (well formed throughout) with
/// distinct valid keys.
pub fn paths(json: &str) -> Option<Vec<String>> {
    let mut s = Scanner {
        b: json.as_bytes(),
        at: 0,
    };
    s.ws();
    s.eat(b'{')?;
    let mut keys: Vec<String> = Vec::new();
    s.ws();
    if s.peek() == Some(b'}') {
        s.at += 1;
    } else {
        loop {
            s.ws();
            let key = s.string()?;
            PartPath::new(&key).ok()?;
            if keys.contains(&key) {
                return None;
            }
            keys.push(key);
            s.ws();
            s.eat(b':')?;
            s.value(0)?;
            s.ws();
            match s.next()? {
                b',' => {}
                b'}' => break,
                _ => return None,
            }
        }
    }
    s.ws();
    (s.at == s.b.len()).then_some(keys)
}

/// Nesting deeper than this is refused rather than recursed into.
const MAX_DEPTH: u32 = 32;

struct Scanner<'a> {
    b: &'a [u8],
    at: usize,
}

impl Scanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.at).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.at += 1;
        Some(c)
    }

    fn eat(&mut self, c: u8) -> Option<()> {
        (self.next()? == c).then_some(())
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let mut v = 0;
        for _ in 0..4 {
            v = v * 16 + char::from(self.next()?).to_digit(16)?;
        }
        Some(v)
    }

    /// A string, decoded.
    fn string(&mut self) -> Option<String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            match self.next()? {
                b'"' => return Some(out),
                b'\\' => {
                    let c = match self.next()? {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                self.eat(b'\\')?;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(code)?
                        }
                        _ => return None,
                    };
                    out.push(c);
                }
                c if c < 0x20 => return None,
                c if c < 0x80 => out.push(char::from(c)),
                _ => {
                    // A multi-byte UTF-8 character: `json` is a `str`, so
                    // it is whole; copy it through.
                    let start = self.at - 1;
                    let mut end = self.at;
                    while self.b.get(end).is_some_and(|b| b & 0xC0 == 0x80) {
                        end += 1;
                    }
                    out.push_str(core::str::from_utf8(&self.b[start..end]).ok()?);
                    self.at = end;
                }
            }
        }
    }

    /// Skips one value, checking it is well formed.
    fn value(&mut self, depth: u32) -> Option<()> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        match self.peek()? {
            b'"' => self.string().map(|_| ()),
            b'{' => {
                self.at += 1;
                self.ws();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Some(());
                }
                loop {
                    self.ws();
                    self.string()?;
                    self.ws();
                    self.eat(b':')?;
                    self.value(depth + 1)?;
                    self.ws();
                    match self.next()? {
                        b',' => {}
                        b'}' => return Some(()),
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.at += 1;
                self.ws();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Some(());
                }
                loop {
                    self.value(depth + 1)?;
                    self.ws();
                    match self.next()? {
                        b',' => {}
                        b']' => return Some(()),
                        _ => return None,
                    }
                }
            }
            b't' => self.word(b"true"),
            b'f' => self.word(b"false"),
            b'n' => self.word(b"null"),
            _ => self.number(),
        }
    }

    fn word(&mut self, w: &[u8]) -> Option<()> {
        let end = self.at + w.len();
        (self.b.get(self.at..end)? == w).then(|| self.at = end)
    }

    /// A JSON number: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    fn number(&mut self) -> Option<()> {
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.next()? {
            b'0' => {}
            b'1'..=b'9' => self.digits(),
            _ => return None,
        }
        if self.peek() == Some(b'.') {
            self.at += 1;
            self.digit()?;
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            self.digit()?;
            self.digits();
        }
        Some(())
    }

    fn digit(&mut self) -> Option<()> {
        self.next()?.is_ascii_digit().then_some(())
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use rlmesh_session_core::spec::normalise::authored_paths;

    use super::*;

    #[test]
    fn the_paths_are_the_normalisers() {
        let maps = [
            r#"{"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
                "hz": 50, "command": "position", "safe": "hold",
                "limits": {"pos": [[-0.6, 0.6], [-0.4, 0.4]]}}}"#,
            r#"{"arm/joints":{"kind":"joints","hz":1e2},"cam":{"kind":"camera","note":"a \"b\" \u00e9 \ud83d\ude00"},"z":[true,false,null,-0.5E-3]}"#,
            "{}",
            " { \"a\" : 1 } ",
        ];
        for map in maps {
            assert_eq!(paths(map), authored_paths(map).ok(), "{map}");
        }
    }

    #[test]
    fn anything_else_is_refused() {
        for bad in [
            "",
            "[]",
            "{",
            "{\"a\":1,}",
            "{\"a\":1} x",
            "{\"a\":01}",
            "{\"a\":1.}",
            "{\"a\":tRue}",
            "{\"a\":\"\x01\"}",
            "{\"a\":1,\"a\":2}",
            "{\"\":1}",
            "{\"a\":[1 2]}",
            "{\"a\":\"\\x\"}",
        ] {
            assert_eq!(paths(bad), None, "{bad:?}");
        }
        let deep = alloc::format!("{{\"a\":{}{}}}", "[".repeat(40), "]".repeat(40));
        assert_eq!(paths(&deep), None);
    }
}
