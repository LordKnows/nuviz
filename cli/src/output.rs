//! Helpers for emitting text formats (LaTeX, CSV, Markdown) safely.

/// Escape text for use in LaTeX body text.
pub fn latex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '&' | '%' | '$' | '#' | '_' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => out.push(c),
        }
    }
    out
}

/// Quote a CSV field when it contains a delimiter, quote or newline (RFC 4180).
pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Join already-raw fields into one CSV line.
pub fn csv_line<I, S>(fields: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    fields
        .into_iter()
        .map(|f| csv_field(f.as_ref()))
        .collect::<Vec<_>>()
        .join(",")
}

/// Escape text for a Markdown table cell.
pub fn markdown_cell(s: &str) -> String {
    s.replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latex_escape_specials() {
        assert_eq!(latex_escape("val_loss"), "val\\_loss");
        assert_eq!(latex_escape("acc@5 (%)"), "acc@5 (\\%)");
        assert_eq!(latex_escape("a&b#c$d"), "a\\&b\\#c\\$d");
        assert_eq!(
            latex_escape("x^2~y"),
            "x\\textasciicircum{}2\\textasciitilde{}y"
        );
        assert_eq!(latex_escape("a\\b"), "a\\textbackslash{}b");
        assert_eq!(latex_escape("{x}"), "\\{x\\}");
    }

    #[test]
    fn test_csv_field_quoting() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_line(["a", "b,c", "d"]), "a,\"b,c\",d");
    }

    #[test]
    fn test_markdown_cell() {
        assert_eq!(markdown_cell("a|b"), "a\\|b");
    }
}
