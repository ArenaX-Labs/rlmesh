/// Values of a TOML array of strings (`key = ["a", "b"]`), written on one line
/// or spread across several — the array ends at the first `]`.
pub fn manifest_string_list(text: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key} = [");
    let mut lines = text.lines().map(str::trim);
    let Some(first) = lines.find_map(|line| line.strip_prefix(prefix.as_str())) else {
        return Vec::new();
    };
    let mut items = String::new();
    for line in std::iter::once(first).chain(lines) {
        let line = line.split_once('#').map_or(line, |(head, _)| head);
        if let Some((head, _)) = line.split_once(']') {
            items.push_str(head);
            break;
        }
        items.push_str(line);
        items.push(',');
    }
    items
        .split(',')
        .filter_map(|item| {
            let (value, _) = item.trim().strip_prefix('"')?.split_once('"')?;
            Some(value.to_string())
        })
        .collect()
}
