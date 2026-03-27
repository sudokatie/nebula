//! Mesh - collection of triangles

use crate::math::{Vec3, Ray};
use crate::accel::{AABB, BVH};
use super::{HitRecord, Hittable, Triangle};

/// A mesh of triangles with its own BVH
pub struct Mesh {
    triangles: Vec<Triangle>,
    bvh: Option<BVH>,
    bounds: AABB,
    material_id: usize,
}

impl Mesh {
    /// Create mesh from triangles
    pub fn new(triangles: Vec<Triangle>, material_id: usize) -> Self {
        let bounds = Self::compute_bounds(&triangles);
        Self {
            triangles,
            bvh: None,
            bounds,
            material_id,
        }
    }

    /// Create mesh from vertices and indices
    pub fn from_indexed(
        vertices: &[Vec3],
        normals: Option<&[Vec3]>,
        indices: &[[usize; 3]],
        material_id: usize,
    ) -> Self {
        Self::from_indexed_with_uvs(vertices, normals, None, indices, material_id)
    }

    /// Create mesh from vertices, normals, UVs, and indices
    pub fn from_indexed_with_uvs(
        vertices: &[Vec3],
        normals: Option<&[Vec3]>,
        uvs: Option<&[(f32, f32)]>,
        indices: &[[usize; 3]],
        material_id: usize,
    ) -> Self {
        let triangles: Vec<Triangle> = indices
            .iter()
            .map(|[i0, i1, i2]| {
                let v0 = vertices[*i0];
                let v1 = vertices[*i1];
                let v2 = vertices[*i2];
                
                let (n0, n1, n2) = if let Some(norms) = normals {
                    (norms[*i0], norms[*i1], norms[*i2])
                } else {
                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let n = edge1.cross(&edge2).normalize();
                    (n, n, n)
                };
                
                let (uv0, uv1, uv2) = if let Some(tex_coords) = uvs {
                    (tex_coords[*i0], tex_coords[*i1], tex_coords[*i2])
                } else {
                    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
                };
                
                Triangle::with_normals_and_uvs(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, material_id)
            })
            .collect();

        Self::new(triangles, material_id)
    }

    /// Build BVH for this mesh
    pub fn build_bvh(&mut self) {
        if self.triangles.is_empty() {
            return;
        }

        let primitives: Vec<Box<dyn Hittable>> = self
            .triangles
            .drain(..)
            .map(|t| Box::new(t) as Box<dyn Hittable>)
            .collect();

        self.bvh = Some(BVH::new(primitives));
    }

    fn compute_bounds(triangles: &[Triangle]) -> AABB {
        let mut bounds = AABB::empty();
        for tri in triangles {
            if let Some(b) = tri.bounding_box() {
                bounds = AABB::surrounding(&bounds, &b);
            }
        }
        bounds
    }

    pub fn triangle_count(&self) -> usize {
        if let Some(bvh) = &self.bvh {
            bvh.primitive_count()
        } else {
            self.triangles.len()
        }
    }

    /// Get the material ID for this mesh
    pub fn material_id(&self) -> usize {
        self.material_id
    }

    /// Get triangles (before BVH is built)
    pub fn triangles(&self) -> &[Triangle] {
        &self.triangles
    }

    /// Check if this mesh has a BVH built
    pub fn has_bvh(&self) -> bool {
        self.bvh.is_some()
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Use BVH if built
        if let Some(bvh) = &self.bvh {
            return bvh.hit(ray, t_min, t_max);
        }

        // Linear search fallback
        let mut closest: Option<HitRecord> = None;
        let mut closest_t = t_max;

        for tri in &self.triangles {
            if let Some(rec) = tri.hit(ray, t_min, closest_t) {
                closest_t = rec.t;
                closest = Some(rec);
            }
        }

        closest
    }

    fn bounding_box(&self) -> Option<AABB> {
        Some(self.bounds)
    }
}

/// Parsed face vertex indices (position, texture, normal)
#[derive(Clone, Copy, Default)]
struct FaceVertex {
    v: usize,
    vt: Option<usize>,
    vn: Option<usize>,
}

/// Load mesh from OBJ file with full support for positions, normals, and UVs
pub fn load_obj(path: &std::path::Path, material_id: usize) -> std::io::Result<Mesh> {
    let content = std::fs::read_to_string(path)?;
    
    let mut positions = Vec::new();
    let mut tex_coords = Vec::new();
    let mut normals = Vec::new();
    let mut faces: Vec<[FaceVertex; 3]> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" if parts.len() >= 4 => {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                positions.push(Vec3::new(x, y, z));
            }
            "vt" if parts.len() >= 3 => {
                let u: f32 = parts[1].parse().unwrap_or(0.0);
                let v: f32 = parts[2].parse().unwrap_or(0.0);
                tex_coords.push((u, v));
            }
            "vn" if parts.len() >= 4 => {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                normals.push(Vec3::new(x, y, z));
            }
            "f" if parts.len() >= 4 => {
                // Parse face vertices (supports v, v/vt, v/vt/vn, v//vn)
                let parse_face_vertex = |s: &str| -> FaceVertex {
                    let components: Vec<&str> = s.split('/').collect();
                    let v = components.first()
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|i| i - 1)
                        .unwrap_or(0);
                    let vt = components.get(1)
                        .filter(|s| !s.is_empty())
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|i| i - 1);
                    let vn = components.get(2)
                        .filter(|s| !s.is_empty())
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|i| i - 1);
                    FaceVertex { v, vt, vn }
                };
                
                let fv0 = parse_face_vertex(parts[1]);
                let fv1 = parse_face_vertex(parts[2]);
                let fv2 = parse_face_vertex(parts[3]);
                faces.push([fv0, fv1, fv2]);

                // Triangulate quads and higher polygons
                for i in 3..(parts.len() - 1) {
                    let fvi = parse_face_vertex(parts[i + 1]);
                    let prev_fv = parse_face_vertex(parts[i]);
                    faces.push([fv0, prev_fv, fvi]);
                }
            }
            _ => {}
        }
    }

    // Build triangles with proper attribute interpolation
    let mut triangles = Vec::with_capacity(faces.len());
    
    for face in &faces {
        let v0 = positions.get(face[0].v).copied().unwrap_or(Vec3::zero());
        let v1 = positions.get(face[1].v).copied().unwrap_or(Vec3::zero());
        let v2 = positions.get(face[2].v).copied().unwrap_or(Vec3::zero());
        
        // Get normals (compute face normal if not provided)
        let face_normal = {
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            e1.cross(&e2).normalize()
        };
        
        let n0 = face[0].vn.and_then(|i| normals.get(i).copied()).unwrap_or(face_normal);
        let n1 = face[1].vn.and_then(|i| normals.get(i).copied()).unwrap_or(face_normal);
        let n2 = face[2].vn.and_then(|i| normals.get(i).copied()).unwrap_or(face_normal);
        
        // Get UVs (use default if not provided)
        let uv0 = face[0].vt.and_then(|i| tex_coords.get(i).copied()).unwrap_or((0.0, 0.0));
        let uv1 = face[1].vt.and_then(|i| tex_coords.get(i).copied()).unwrap_or((1.0, 0.0));
        let uv2 = face[2].vt.and_then(|i| tex_coords.get(i).copied()).unwrap_or((0.0, 1.0));
        
        triangles.push(Triangle::with_normals_and_uvs(
            v0, v1, v2,
            n0, n1, n2,
            uv0, uv1, uv2,
            material_id,
        ));
    }

    let mut mesh = Mesh::new(triangles, material_id);
    mesh.build_bvh();
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_from_triangles() {
        let tris = vec![
            Triangle::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0,
            ),
        ];
        let mesh = Mesh::new(tris, 0);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_mesh_from_indexed() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![[0, 1, 2]];
        let mesh = Mesh::from_indexed(&verts, None, &indices, 0);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_mesh_hit() {
        let verts = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![[0, 1, 2]];
        let mesh = Mesh::from_indexed(&verts, None, &indices, 0);
        
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(mesh.hit(&ray, 0.001, f32::INFINITY).is_some());
    }

    #[test]
    fn test_mesh_with_bvh() {
        let verts = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, -2.0),
            Vec3::new(1.0, -1.0, -2.0),
            Vec3::new(0.0, 1.0, -2.0),
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mut mesh = Mesh::from_indexed(&verts, None, &indices, 0);
        mesh.build_bvh();
        
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = mesh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        assert!((hit.unwrap().t - 1.0).abs() < 0.01);
    }
}
