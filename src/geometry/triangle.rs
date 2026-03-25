//! Triangle primitive with Moller-Trumbore intersection

use crate::math::{Vec3, Ray};
use crate::accel::AABB;
use super::{HitRecord, Hittable};

/// A triangle primitive
#[derive(Debug, Clone)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub material_id: usize,
}

impl Triangle {
    /// Create triangle with flat shading (computed normal)
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material_id: usize) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2).normalize();
        Self {
            v0, v1, v2,
            n0: normal,
            n1: normal,
            n2: normal,
            material_id,
        }
    }

    /// Create triangle with vertex normals (smooth shading)
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
            material_id,
        }
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
        
        // Interpolate normal
        let outward_normal = (self.n0 * w + self.n1 * u + self.n2 * v).normalize();

        let mut rec = HitRecord {
            t,
            point,
            normal: Vec3::zero(),
            uv: (u, v),
            front_face: false,
            material_id: self.material_id,
        };
        rec.set_face_normal(ray, outward_normal);

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
