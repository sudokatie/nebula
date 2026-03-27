//! OBJ mesh file loader

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::math::Vec3;
use crate::geometry::{Mesh, Triangle};

/// Load a mesh from an OBJ file
pub fn load_obj(path: &Path, material_id: usize) -> io::Result<Mesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<(f32, f32)> = Vec::new();
    let mut triangles: Vec<Triangle> = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" => {
                // Vertex position
                if parts.len() >= 4 {
                    let x = parts[1].parse::<f32>().unwrap_or(0.0);
                    let y = parts[2].parse::<f32>().unwrap_or(0.0);
                    let z = parts[3].parse::<f32>().unwrap_or(0.0);
                    vertices.push(Vec3::new(x, y, z));
                }
            }
            "vn" => {
                // Vertex normal
                if parts.len() >= 4 {
                    let x = parts[1].parse::<f32>().unwrap_or(0.0);
                    let y = parts[2].parse::<f32>().unwrap_or(0.0);
                    let z = parts[3].parse::<f32>().unwrap_or(0.0);
                    normals.push(Vec3::new(x, y, z));
                }
            }
            "vt" => {
                // Texture coordinate
                if parts.len() >= 3 {
                    let u = parts[1].parse::<f32>().unwrap_or(0.0);
                    let v = parts[2].parse::<f32>().unwrap_or(0.0);
                    uvs.push((u, v));
                }
            }
            "f" => {
                // Face (triangulate if needed)
                let face_verts = parse_face(&parts[1..], &vertices, &normals);
                if face_verts.len() >= 3 {
                    // Triangulate polygon (fan triangulation)
                    for i in 1..(face_verts.len() - 1) {
                        let (v0, n0) = face_verts[0];
                        let (v1, n1) = face_verts[i];
                        let (v2, n2) = face_verts[i + 1];

                        let tri = if let (Some(n0), Some(n1), Some(n2)) = (n0, n1, n2) {
                            Triangle::with_normals(v0, v1, v2, n0, n1, n2, material_id)
                        } else {
                            Triangle::new(v0, v1, v2, material_id)
                        };
                        triangles.push(tri);
                    }
                }
            }
            _ => {
                // Ignore other directives (mtllib, usemtl, s, o, g, etc.)
            }
        }
    }

    if triangles.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No triangles found in OBJ file",
        ));
    }

    Ok(Mesh::new(triangles, material_id))
}

/// Parse face vertex indices
fn parse_face(
    parts: &[&str],
    vertices: &[Vec3],
    normals: &[Vec3],
) -> Vec<(Vec3, Option<Vec3>)> {
    let mut face_verts = Vec::with_capacity(parts.len());

    for part in parts {
        let indices: Vec<&str> = part.split('/').collect();
        if indices.is_empty() {
            continue;
        }

        // Parse vertex index (1-based in OBJ)
        let v_idx = indices[0]
            .parse::<i32>()
            .map(|i| if i < 0 { vertices.len() as i32 + i } else { i - 1 })
            .ok();

        // Parse normal index if present
        let n_idx = if indices.len() >= 3 && !indices[2].is_empty() {
            indices[2]
                .parse::<i32>()
                .map(|i| if i < 0 { normals.len() as i32 + i } else { i - 1 })
                .ok()
        } else {
            None
        };

        if let Some(vi) = v_idx {
            let vi = vi as usize;
            if vi < vertices.len() {
                let vertex = vertices[vi];
                let normal = n_idx.and_then(|ni| {
                    let ni = ni as usize;
                    if ni < normals.len() {
                        Some(normals[ni])
                    } else {
                        None
                    }
                });
                face_verts.push((vertex, normal));
            }
        }
    }

    face_verts
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_obj(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    #[test]
    fn test_load_simple_triangle() {
        let content = r#"
# Simple triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"#;
        let file = create_test_obj(content);
        let mesh = load_obj(file.path(), 0).unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_load_quad_face() {
        let content = r#"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"#;
        let file = create_test_obj(content);
        let mesh = load_obj(file.path(), 0).unwrap();
        // Quad should be triangulated into 2 triangles
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_load_with_normals() {
        let content = r#"
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
"#;
        let file = create_test_obj(content);
        let mesh = load_obj(file.path(), 0).unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_load_with_uvs() {
        let content = r#"
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"#;
        let file = create_test_obj(content);
        let mesh = load_obj(file.path(), 0).unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_load_empty_file() {
        let content = "# Empty file\n";
        let file = create_test_obj(content);
        let result = load_obj(file.path(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_indices() {
        let content = r#"
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
"#;
        let file = create_test_obj(content);
        let mesh = load_obj(file.path(), 0).unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }
}
