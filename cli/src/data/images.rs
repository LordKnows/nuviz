use std::path::{Path, PathBuf};

/// An image file discovered in an experiment's images/ directory.
#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub step: u64,
    pub tag: String,
    pub size_bytes: u64,
}

/// Discover all images in an experiment directory, sorted by step then tag.
///
/// Expected filename pattern: `step_NNNNNN_<tag>.png`
pub fn discover_images(experiment_dir: &Path) -> Vec<ImageEntry> {
    let images_dir = experiment_dir.join("images");
    if !images_dir.is_dir() {
        return Vec::new();
    }

    let mut entries = Vec::new();

    let read_dir = match std::fs::read_dir(&images_dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }

        if let Some(parsed) = parse_image_filename(&path) {
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            entries.push(ImageEntry {
                path,
                step: parsed.0,
                tag: parsed.1,
                size_bytes,
            });
        }
    }

    entries.sort_by(|a, b| a.step.cmp(&b.step).then_with(|| a.tag.cmp(&b.tag)));
    entries
}

/// Find the latest image matching an optional tag filter.
pub fn find_latest_image(experiment_dir: &Path, tag: Option<&str>) -> Option<ImageEntry> {
    let mut images = discover_images(experiment_dir);
    if let Some(tag_filter) = tag {
        images.retain(|e| e.tag == tag_filter);
    }
    images.into_iter().last()
}

/// Parse `step_NNNNNN_<tag>.png` -> (step, tag)
fn parse_image_filename(path: &Path) -> Option<(u64, String)> {
    let stem = path.file_stem()?.to_str()?;
    let parts: Vec<&str> = stem.splitn(3, '_').collect();
    if parts.len() < 3 || parts[0] != "step" {
        return None;
    }
    let step = parts[1].parse::<u64>().ok()?;
    let tag = parts[2].to_string();
    Some((step, tag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_parse_image_filename() {
        let path = Path::new("/tmp/images/step_000500_render.png");
        let (step, tag) = parse_image_filename(path).unwrap();
        assert_eq!(step, 500);
        assert_eq!(tag, "render");
    }

    #[test]
    fn test_parse_image_filename_with_underscores_in_tag() {
        let path = Path::new("/tmp/images/step_001000_depth_map.png");
        let (step, tag) = parse_image_filename(path).unwrap();
        assert_eq!(step, 1000);
        assert_eq!(tag, "depth_map");
    }

    #[test]
    fn test_parse_image_filename_invalid() {
        assert!(parse_image_filename(Path::new("/tmp/images/random.png")).is_none());
        assert!(parse_image_filename(Path::new("/tmp/images/step_abc_tag.png")).is_none());
    }

    #[test]
    fn test_discover_images() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        fs::create_dir_all(&images_dir).unwrap();

        // Create test image files (just empty files for discovery)
        fs::write(images_dir.join("step_000000_render.png"), b"fake").unwrap();
        fs::write(images_dir.join("step_000000_gt.png"), b"fake").unwrap();
        fs::write(images_dir.join("step_000100_render.png"), b"fake").unwrap();
        fs::write(images_dir.join("not_an_image.txt"), b"nope").unwrap();

        let entries = discover_images(dir.path());
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].step, 0);
        assert_eq!(entries[0].tag, "gt");
        assert_eq!(entries[1].step, 0);
        assert_eq!(entries[1].tag, "render");
        assert_eq!(entries[2].step, 100);
    }

    #[test]
    fn test_discover_images_empty() {
        let dir = tempfile::tempdir().unwrap();
        let entries = discover_images(dir.path());
        assert!(entries.is_empty());
    }

    #[test]
    fn test_find_latest_image() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        fs::create_dir_all(&images_dir).unwrap();

        fs::write(images_dir.join("step_000000_render.png"), b"fake").unwrap();
        fs::write(images_dir.join("step_000100_render.png"), b"fake").unwrap();
        fs::write(images_dir.join("step_000100_depth.png"), b"fake").unwrap();

        let latest = find_latest_image(dir.path(), Some("render")).unwrap();
        assert_eq!(latest.step, 100);
        assert_eq!(latest.tag, "render");

        let latest_any = find_latest_image(dir.path(), None).unwrap();
        assert_eq!(latest_any.step, 100);
    }
}
