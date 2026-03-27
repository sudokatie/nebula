//! Mesh primitive (collection of triangles with BVH)

use crate::math::{Vec3, Ray};
use crate::accel::{AABB, BVH};
use super::{HitRecord, Hittable, Triangle};

/// A mesh consisting of multiple triangles
pub struct Mesh {
    bvh: BVH,
    bounds: AABB,
    material_id: usize,
}

impl Mesh {
    /// Create mesh from triangles
    pub fn new(triangles: Vec<Triangle>, material_id: usize) -> Self {
        // Compute overall bounds
        let mut bounds = AABB::empty();
        for tri in &triangles {
            if let Some(b) = tri.bounding_box() {
                bounds = AABB::surrounding(&bounds, &b);
            }
        }

        // Build BVH from triangles
        let primitives: Vec<Box<dyn Hittable>> = triangles
            .into_iter()
            .map(|t| Box::new(t) as Box<dyn Hittable>)
            .collect();

        let bvh = BVH::new(primitives);

        Self {
            bvh,
            bounds,
            material_id,
        }
    }

    /// Create mesh from indexed vertex data
    pub fn from_indexed(
        vertices: &[Vec3],
        normals: Option<&[Vec3]>,
        uvs: Option<&[(f32, f32)]>,
        indices: &[[usize; 3]],
        material_id: usize,
    ) -> Self {
        let mut triangles = Vec::with_capacity(indices.len());

        for &[i0, i1, i2] in indices {
            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];

            let triangle = if let Some(norms) = normals {
                Triangle::with_normals(
                    v0, v1, v2,
                    norms[i0], norms[i1], norms[i2],
                    material_id,
                )
            } else {
                Triangle::new(v0, v1, v2, material_id)
            };

            // TODO: Use UVs when texture support is added
            let _ = uvs;

            triangles.push(triangle);
        }

        Self::new(triangles, material_id)
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.bvh.primitive_count()
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        self.bvh.hit(ray, t_min, t_max)
    }

    fn bounding_box(&self) -> Option<AABB> {
        Some(self.bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_from_triangles() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0,
            ),
            Triangle::new(
                Vec3::new(-1.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
                0,
            ),
        ];

        let mesh = Mesh::new(triangles, 0);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_mesh_hit() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0,
            ),
        ];

        let mesh = Mesh::new(triangles, 0);
        let ray = Ray::new(Vec3::new(0.0, 0.3, 1.0), Vec3::new(0.0, 0.0, -1.0));

        let hit = mesh.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_mesh_from_indexed() {
        let vertices = vec![
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let indices = vec![[0, 1, 2], [0, 1, 3]];

        let mesh = Mesh::from_indexed(&vertices, None, None, &indices, 0);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_mesh_bounding_box() {
        let triangles = vec![
            Triangle::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0,
            ),
        ];

        let mesh = Mesh::new(triangles, 0);
        let bounds = mesh.bounding_box().unwrap();

        assert!(bounds.min.x <= -1.0);
        assert!(bounds.max.x >= 1.0);
        assert!(bounds.max.y >= 1.0);
    }
}
