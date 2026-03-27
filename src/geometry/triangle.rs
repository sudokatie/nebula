//! Triangle primitive with Moller-Trumbore intersection

use crate::math::{Vec3, Ray};
use crate::accel::AABB;
use super::{HitRecord, Hittable};

/// A triangle primitive with vertex positions, normals, and UVs
#[derive(Debug, Clone)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub uv0: (f32, f32),
    pub uv1: (f32, f32),
    pub uv2: (f32, f32),
    pub material_id: usize,
}

impl Triangle {
    /// Create triangle with flat shading (computed normal) and default UVs
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material_id: usize) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2).normalize();
        Self {
            v0, v1, v2,
            n0: normal,
            n1: normal,
            n2: normal,
            uv0: (0.0, 0.0),
            uv1: (1.0, 0.0),
            uv2: (0.0, 1.0),
            material_id,
        }
    }

    /// Create triangle with vertex normals (smooth shading) and default UVs
    pub fn with_normals(
        v0: Vec3, v1: Vec3, v2: Vec3,
        n0: Vec3, n1: Vec3, n2: Vec3,
        material_id: usize,
    ) -> Self {
        Self {
            v0, v1, v2,
            n0: n0.normalize(),
            n1: n1.normalize(),
            n2: n2.normalize(),
            uv0: (0.0, 0.0),
            uv1: (1.0, 0.0),
            uv2: (0.0, 1.0),
            material_id,
        }
    }

    /// Create triangle with vertex normals and UVs
    pub fn with_normals_and_uvs(
        v0: Vec3, v1: Vec3, v2: Vec3,
        n0: Vec3, n1: Vec3, n2: Vec3,
        uv0: (f32, f32), uv1: (f32, f32), uv2: (f32, f32),
        material_id: usize,
    ) -> Self {
        Self {
            v0, v1, v2,
            n0: n0.normalize(),
            n1: n1.normalize(),
            n2: n2.normalize(),
            uv0, uv1, uv2,
            material_id,
        }
    }

    /// Create triangle with UVs only (flat shading)
    pub fn with_uvs(
        v0: Vec3, v1: Vec3, v2: Vec3,
        uv0: (f32, f32), uv1: (f32, f32), uv2: (f32, f32),
        material_id: usize,
    ) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2).normalize();
        Self {
            v0, v1, v2,
            n0: normal,
            n1: normal,
            n2: normal,
            uv0, uv1, uv2,
            material_id,
        }
    }

    /// Compute the area of this triangle
    pub fn area(&self) -> f32 {
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        edge1.cross(&edge2).length() * 0.5
    }

    /// Get the geometric (face) normal
    pub fn face_normal(&self) -> Vec3 {
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        edge1.cross(&edge2).normalize()
    }

    /// Get the centroid of this triangle
    pub fn centroid(&self) -> Vec3 {
        (self.v0 + self.v1 + self.v2) / 3.0
    }

    /// Sample a random point on the triangle surface (uniform)
    pub fn sample_point(&self, u1: f32, u2: f32) -> (Vec3, Vec3) {
        // Use square-to-triangle mapping
        let su0 = u1.sqrt();
        let b0 = 1.0 - su0;
        let b1 = u2 * su0;
        let b2 = 1.0 - b0 - b1;
        
        let point = self.v0 * b0 + self.v1 * b1 + self.v2 * b2;
        let normal = (self.n0 * b0 + self.n1 * b1 + self.n2 * b2).normalize();
        (point, normal)
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        const EPSILON: f32 = 1e-8;

        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let h = ray.direction.cross(&edge2);
        let a = edge1.dot(&h);

        // Ray parallel to triangle
        if a.abs() < EPSILON {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(&h);

        // Outside triangle
        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * ray.direction.dot(&q);

        // Outside triangle
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(&q);

        // Check range
        if t <= t_min || t >= t_max {
            return None;
        }

        let w = 1.0 - u - v;
        let point = ray.at(t);
        
        // Geometric normal (from triangle edges)
        let geometric_normal = edge1.cross(&edge2).normalize();
        
        // Shading normal (interpolated from vertex normals)
        let shading_normal = (self.n0 * w + self.n1 * u + self.n2 * v).normalize();

        // Interpolate UV coordinates
        let tex_u = self.uv0.0 * w + self.uv1.0 * u + self.uv2.0 * v;
        let tex_v = self.uv0.1 * w + self.uv1.1 * u + self.uv2.1 * v;

        let mut rec = HitRecord {
            t,
            point,
            geometric_normal: Vec3::zero(),
            normal: Vec3::zero(),
            uv: (tex_u, tex_v),
            front_face: false,
            material_id: self.material_id,
        };
        rec.set_normals(ray, geometric_normal, shading_normal);

        Some(rec)
    }

    fn bounding_box(&self) -> Option<AABB> {
        let min = self.v0.min(&self.v1).min(&self.v2);
        let max = self.v0.max(&self.v1).max(&self.v2);
        
        // Add small epsilon to avoid zero-thickness boxes
        let epsilon = Vec3::new(0.0001, 0.0001, 0.0001);
        Some(AABB::new(min - epsilon, max + epsilon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_hit() {
        let tri = Triangle::new(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );
        let ray = Ray::new(Vec3::new(0.0, 0.3, 1.0), Vec3::new(0.0, 0.0, -1.0));
        
        let hit = tri.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn test_triangle_miss() {
        let tri = Triangle::new(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );
        let ray = Ray::new(Vec3::new(5.0, 5.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        
        let hit = tri.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_none());
    }
}
