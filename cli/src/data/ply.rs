use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Parsed PLY file data.
#[derive(Debug)]
pub struct PlyData {
    pub num_vertices: usize,
    pub positions: Vec<[f32; 3]>,
    pub colors: Option<Vec<[u8; 3]>>,
    pub opacities: Option<Vec<f32>>,
    pub scales: Option<Vec<[f32; 3]>>,
    #[allow(dead_code)]
    pub rotations: Option<Vec<[f32; 4]>>,
    pub sh_degree: Option<u32>,
    pub custom_properties: HashMap<String, Vec<f32>>,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

#[derive(Debug, Clone)]
struct PropertyDef {
    name: String,
    dtype: PropertyType,
}

#[derive(Debug, Clone, Copy)]
enum PropertyType {
    Float32,
    Float64,
    UChar,
    Int32,
    UInt32,
    Int16,
    UInt16,
    Int8,
}

impl PropertyType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "float" | "float32" => Some(Self::Float32),
            "double" | "float64" => Some(Self::Float64),
            "uchar" | "uint8" => Some(Self::UChar),
            "int" | "int32" => Some(Self::Int32),
            "uint" | "uint32" => Some(Self::UInt32),
            "short" | "int16" => Some(Self::Int16),
            "ushort" | "uint16" => Some(Self::UInt16),
            "char" | "int8" => Some(Self::Int8),
            _ => None,
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::UInt32 => 4,
            Self::Float64 => 8,
            Self::Int16 | Self::UInt16 => 2,
            Self::UChar | Self::Int8 => 1,
        }
    }
}

/// Parse a PLY file from the given path.
pub fn parse_ply(path: &Path) -> Result<PlyData> {
    let file_size_bytes = std::fs::metadata(path)
        .with_context(|| format!("Cannot read PLY file: {}", path.display()))?
        .len();

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Parse header
    let (format, num_vertices, properties) = parse_header(&mut reader)?;

    // Parse vertex data
    let raw = match format {
        PlyFormat::BinaryLittleEndian => {
            read_binary_vertices(&mut reader, num_vertices, &properties)?
        }
        PlyFormat::Ascii => read_ascii_vertices(&mut reader, num_vertices, &properties)?,
        PlyFormat::BinaryBigEndian => bail!("Binary big-endian PLY is not supported"),
    };

    // Extract structured fields
    let ply = extract_fields(raw, &properties, num_vertices, file_size_bytes)?;
    Ok(ply)
}

/// Parse PLY header, returning format, vertex count, and property definitions.
fn parse_header(reader: &mut BufReader<File>) -> Result<(PlyFormat, usize, Vec<PropertyDef>)> {
    let mut line = String::new();
    let mut format = None;
    let mut num_vertices = 0usize;
    let mut properties = Vec::new();
    let mut in_vertex_element = false;

    // Verify magic number
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        bail!("Not a PLY file (missing 'ply' magic)");
    }

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            bail!("Unexpected EOF in PLY header (missing 'end_header')");
        }

        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                if parts.len() < 3 {
                    bail!("Invalid format line: {trimmed}");
                }
                format = Some(match parts[1] {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    other => bail!("Unknown PLY format: {other}"),
                });
            }
            "element" => {
                if parts.len() >= 3 && parts[1] == "vertex" {
                    num_vertices = parts[2].parse().context("Invalid vertex count")?;
                    in_vertex_element = true;
                } else {
                    in_vertex_element = false;
                }
            }
            "property" => {
                if !in_vertex_element {
                    continue;
                }
                // Skip list properties (e.g., face vertex_indices)
                if parts.len() >= 2 && parts[1] == "list" {
                    continue;
                }
                if parts.len() < 3 {
                    continue;
                }
                let dtype = PropertyType::from_str(parts[1]);
                if let Some(dt) = dtype {
                    properties.push(PropertyDef {
                        name: parts[2].to_string(),
                        dtype: dt,
                    });
                }
            }
            _ => {} // Ignore comments, obj_info, etc.
        }
    }

    let format = format.ok_or_else(|| anyhow::anyhow!("No format declaration in PLY header"))?;

    if num_vertices == 0 {
        bail!("No vertex element found in PLY header");
    }

    Ok((format, num_vertices, properties))
}

/// Read binary vertex data into a flat Vec of f32 per property per vertex.
fn read_binary_vertices(
    reader: &mut BufReader<File>,
    num_vertices: usize,
    properties: &[PropertyDef],
) -> Result<Vec<Vec<f32>>> {
    let row_size: usize = properties.iter().map(|p| p.dtype.byte_size()).sum();
    let total_bytes = row_size * num_vertices;

    let mut raw_buf = vec![0u8; total_bytes];
    reader.read_exact(&mut raw_buf)?;

    let mut columns: Vec<Vec<f32>> = properties
        .iter()
        .map(|_| Vec::with_capacity(num_vertices))
        .collect();

    let mut offset = 0;
    for _ in 0..num_vertices {
        for (col_idx, prop) in properties.iter().enumerate() {
            let value = read_value_le(&raw_buf[offset..], prop.dtype);
            columns[col_idx].push(value);
            offset += prop.dtype.byte_size();
        }
    }

    Ok(columns)
}

/// Read a single value from a byte slice in little-endian format, returning as f32.
fn read_value_le(buf: &[u8], dtype: PropertyType) -> f32 {
    match dtype {
        PropertyType::Float32 => f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        PropertyType::Float64 => f64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]) as f32,
        PropertyType::UChar => buf[0] as f32,
        PropertyType::Int8 => buf[0] as i8 as f32,
        PropertyType::Int16 => i16::from_le_bytes([buf[0], buf[1]]) as f32,
        PropertyType::UInt16 => u16::from_le_bytes([buf[0], buf[1]]) as f32,
        PropertyType::Int32 => i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
        PropertyType::UInt32 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f32,
    }
}

/// Read ASCII vertex data.
fn read_ascii_vertices(
    reader: &mut BufReader<File>,
    num_vertices: usize,
    properties: &[PropertyDef],
) -> Result<Vec<Vec<f32>>> {
    let mut columns: Vec<Vec<f32>> = properties
        .iter()
        .map(|_| Vec::with_capacity(num_vertices))
        .collect();
    let mut line = String::new();

    for row_idx in 0..num_vertices {
        line.clear();
        reader.read_line(&mut line)?;
        let values: Vec<&str> = line.split_whitespace().collect();

        if values.len() < properties.len() {
            bail!(
                "ASCII PLY row {} has {} values, expected {}",
                row_idx,
                values.len(),
                properties.len()
            );
        }

        for (col_idx, prop) in properties.iter().enumerate() {
            let v: f32 = match prop.dtype {
                PropertyType::UChar | PropertyType::Int8 => {
                    values[col_idx].parse::<i32>().unwrap_or(0) as f32
                }
                _ => values[col_idx].parse().unwrap_or(0.0),
            };
            columns[col_idx].push(v);
        }
    }

    Ok(columns)
}

/// Extract structured fields from raw column data.
fn extract_fields(
    columns: Vec<Vec<f32>>,
    properties: &[PropertyDef],
    num_vertices: usize,
    file_size_bytes: u64,
) -> Result<PlyData> {
    let prop_index: HashMap<&str, usize> = properties
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.as_str(), i))
        .collect();

    // Positions (required)
    let x_idx = prop_index
        .get("x")
        .ok_or_else(|| anyhow::anyhow!("Missing 'x' property"))?;
    let y_idx = prop_index
        .get("y")
        .ok_or_else(|| anyhow::anyhow!("Missing 'y' property"))?;
    let z_idx = prop_index
        .get("z")
        .ok_or_else(|| anyhow::anyhow!("Missing 'z' property"))?;

    let positions: Vec<[f32; 3]> = columns[*x_idx]
        .iter()
        .zip(columns[*y_idx].iter())
        .zip(columns[*z_idx].iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    // Colors (optional)
    let colors = if let (Some(&ri), Some(&gi), Some(&bi)) = (
        prop_index.get("red"),
        prop_index.get("green"),
        prop_index.get("blue"),
    ) {
        let v: Vec<[u8; 3]> = columns[ri]
            .iter()
            .zip(columns[gi].iter())
            .zip(columns[bi].iter())
            .map(|((&r, &g), &b)| [r as u8, g as u8, b as u8])
            .collect();
        Some(v)
    } else {
        None
    };

    // Opacities
    let opacities = prop_index.get("opacity").map(|&idx| columns[idx].clone());

    // Scales (3DGS)
    let scales = if let (Some(&s0), Some(&s1), Some(&s2)) = (
        prop_index.get("scale_0"),
        prop_index.get("scale_1"),
        prop_index.get("scale_2"),
    ) {
        let v: Vec<[f32; 3]> = columns[s0]
            .iter()
            .zip(columns[s1].iter())
            .zip(columns[s2].iter())
            .map(|((&a, &b), &c)| [a, b, c])
            .collect();
        Some(v)
    } else {
        None
    };

    // Rotations (3DGS quaternion)
    let rotations = if let (Some(&r0), Some(&r1), Some(&r2), Some(&r3)) = (
        prop_index.get("rot_0"),
        prop_index.get("rot_1"),
        prop_index.get("rot_2"),
        prop_index.get("rot_3"),
    ) {
        let v: Vec<[f32; 4]> = columns[r0]
            .iter()
            .zip(columns[r1].iter())
            .zip(columns[r2].iter())
            .zip(columns[r3].iter())
            .map(|(((&a, &b), &c), &d)| [a, b, c, d])
            .collect();
        Some(v)
    } else {
        None
    };

    // SH degree: inferred from f_rest_* count
    // degree 0: 0 rest, degree 1: 9, degree 2: 24, degree 3: 45
    let sh_rest_count = prop_index
        .keys()
        .filter(|k| k.starts_with("f_rest_"))
        .count();
    let sh_degree = match sh_rest_count {
        0 => {
            if prop_index.contains_key("f_dc_0") {
                Some(0)
            } else {
                None
            }
        }
        n if n <= 9 => Some(1),
        n if n <= 24 => Some(2),
        _ => Some(3),
    };

    // Collect remaining properties as custom
    let known_props = [
        "x", "y", "z", "red", "green", "blue", "opacity", "scale_0", "scale_1", "scale_2", "rot_0",
        "rot_1", "rot_2", "rot_3", "nx", "ny", "nz", "alpha",
    ];
    let mut custom_properties = HashMap::new();
    for (i, prop) in properties.iter().enumerate() {
        if !known_props.contains(&prop.name.as_str())
            && !prop.name.starts_with("f_dc_")
            && !prop.name.starts_with("f_rest_")
        {
            custom_properties.insert(prop.name.clone(), columns[i].clone());
        }
    }

    Ok(PlyData {
        num_vertices,
        positions,
        colors,
        opacities,
        scales,
        rotations,
        sh_degree,
        custom_properties,
        file_size_bytes,
    })
}

/// Compute statistics for a PLY file without loading all data into memory.
/// For now delegates to parse_ply; streaming version can be added later.
pub fn compute_ply_stats(path: &Path) -> Result<PlyStats> {
    let ply = parse_ply(path)?;

    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];

    for pos in &ply.positions {
        for i in 0..3 {
            min_pos[i] = min_pos[i].min(pos[i]);
            max_pos[i] = max_pos[i].max(pos[i]);
        }
    }

    let opacity_stats = ply.opacities.as_ref().map(|ops| {
        let (sum, sum_sq, min, max) = ops.iter().fold(
            (0.0f64, 0.0f64, f32::MAX, f32::MIN),
            |(s, sq, mn, mx), &v| (s + v as f64, sq + v as f64 * v as f64, mn.min(v), mx.max(v)),
        );
        let n = ops.len() as f64;
        let mean = sum / n;
        let std = ((sum_sq / n) - mean * mean).max(0.0).sqrt();
        let near_transparent = ops.iter().filter(|&&v| v < 0.01).count();
        AttributeStats {
            mean: mean as f32,
            std: std as f32,
            min,
            max,
            special_count: near_transparent,
        }
    });

    let scale_stats = ply.scales.as_ref().map(|scales| {
        let magnitudes: Vec<f32> = scales
            .iter()
            .map(|s| (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt())
            .collect();
        let (sum, sum_sq, min, max) = magnitudes.iter().fold(
            (0.0f64, 0.0f64, f32::MAX, f32::MIN),
            |(s, sq, mn, mx), &v| (s + v as f64, sq + v as f64 * v as f64, mn.min(v), mx.max(v)),
        );
        let n = magnitudes.len() as f64;
        let mean = sum / n;
        let std = ((sum_sq / n) - mean * mean).max(0.0).sqrt();
        let outliers = magnitudes
            .iter()
            .filter(|&&v| (v as f64 - mean).abs() > 3.0 * std)
            .count();
        AttributeStats {
            mean: mean as f32,
            std: std as f32,
            min,
            max,
            special_count: outliers,
        }
    });

    Ok(PlyStats {
        num_vertices: ply.num_vertices,
        bounding_box: (min_pos, max_pos),
        sh_degree: ply.sh_degree,
        has_colors: ply.colors.is_some(),
        opacity_stats,
        scale_stats,
        file_size_bytes: ply.file_size_bytes,
        custom_property_count: ply.custom_properties.len(),
        opacities: ply.opacities,
        scales: ply.scales,
    })
}

/// Summary statistics for a PLY file.
#[derive(Debug)]
pub struct PlyStats {
    pub num_vertices: usize,
    pub bounding_box: ([f32; 3], [f32; 3]),
    pub sh_degree: Option<u32>,
    pub has_colors: bool,
    pub opacity_stats: Option<AttributeStats>,
    pub scale_stats: Option<AttributeStats>,
    pub file_size_bytes: u64,
    pub custom_property_count: usize,
    /// Raw opacity values (for histogram rendering)
    pub opacities: Option<Vec<f32>>,
    /// Raw scale values (for histogram rendering)
    pub scales: Option<Vec<[f32; 3]>>,
}

#[derive(Debug)]
pub struct AttributeStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    /// Context-dependent: near-transparent count for opacity, outlier count for scale
    pub special_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_ascii_ply(dir: &Path, vertices: &[[f32; 3]]) -> std::path::PathBuf {
        let path = dir.join("test.ply");
        let mut f = File::create(&path).unwrap();

        writeln!(f, "ply").unwrap();
        writeln!(f, "format ascii 1.0").unwrap();
        writeln!(f, "element vertex {}", vertices.len()).unwrap();
        writeln!(f, "property float x").unwrap();
        writeln!(f, "property float y").unwrap();
        writeln!(f, "property float z").unwrap();
        writeln!(f, "end_header").unwrap();

        for v in vertices {
            writeln!(f, "{} {} {}", v[0], v[1], v[2]).unwrap();
        }

        path
    }

    fn write_binary_ply(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("test_bin.ply");
        let mut f = File::create(&path).unwrap();

        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nend_header\n";
        f.write_all(header.as_bytes()).unwrap();

        // 3 vertices: (x,y,z,r,g,b,opacity)
        for i in 0..3u32 {
            let x = i as f32;
            let y = (i as f32) * 2.0;
            let z = (i as f32) * 3.0;
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&y.to_le_bytes()).unwrap();
            f.write_all(&z.to_le_bytes()).unwrap();
            f.write_all(&[255u8, 128, 64]).unwrap(); // RGB
            let opacity = 0.5f32 + i as f32 * 0.2;
            f.write_all(&opacity.to_le_bytes()).unwrap();
        }

        path
    }

    #[test]
    fn test_parse_ascii_ply() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_ascii_ply(
            dir.path(),
            &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );

        let ply = parse_ply(&path).unwrap();
        assert_eq!(ply.num_vertices, 3);
        assert_eq!(ply.positions.len(), 3);
        assert!((ply.positions[0][0] - 1.0).abs() < f32::EPSILON);
        assert!((ply.positions[2][2] - 9.0).abs() < f32::EPSILON);
        assert!(ply.colors.is_none());
        assert!(ply.opacities.is_none());
    }

    #[test]
    fn test_parse_binary_ply() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_binary_ply(dir.path());

        let ply = parse_ply(&path).unwrap();
        assert_eq!(ply.num_vertices, 3);
        assert!(ply.colors.is_some());
        assert_eq!(ply.colors.as_ref().unwrap()[0], [255, 128, 64]);
        assert!(ply.opacities.is_some());
        assert!((ply.opacities.as_ref().unwrap()[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_ply_missing_file() {
        let result = parse_ply(Path::new("/nonexistent/test.ply"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ply_not_ply() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.ply");
        std::fs::write(&path, "not a ply file").unwrap();
        let result = parse_ply(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ply_no_end_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noend.ply");
        std::fs::write(
            &path,
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n",
        )
        .unwrap();
        let result = parse_ply(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_sh_degree_detection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sh.ply");
        let mut f = File::create(&path).unwrap();

        writeln!(f, "ply").unwrap();
        writeln!(f, "format ascii 1.0").unwrap();
        writeln!(f, "element vertex 1").unwrap();
        writeln!(f, "property float x").unwrap();
        writeln!(f, "property float y").unwrap();
        writeln!(f, "property float z").unwrap();
        writeln!(f, "property float f_dc_0").unwrap();
        writeln!(f, "property float f_dc_1").unwrap();
        writeln!(f, "property float f_dc_2").unwrap();
        for i in 0..9 {
            writeln!(f, "property float f_rest_{i}").unwrap();
        }
        writeln!(f, "end_header").unwrap();
        // 1 vertex: x y z f_dc_0..2 f_rest_0..8
        writeln!(f, "0 0 0 0.1 0.2 0.3 0 0 0 0 0 0 0 0 0").unwrap();

        let ply = parse_ply(&path).unwrap();
        assert_eq!(ply.sh_degree, Some(1));
    }

    #[test]
    fn test_compute_ply_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_binary_ply(dir.path());

        let stats = compute_ply_stats(&path).unwrap();
        assert_eq!(stats.num_vertices, 3);
        assert!(stats.has_colors);
        assert!(stats.opacity_stats.is_some());
    }
}
