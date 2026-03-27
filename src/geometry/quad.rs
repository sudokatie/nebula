//! Quad (parallelogram) primitive

use crate::math::{Vec3, Ray};
use crate::accel::AABB;
use super::{HitRecord, Hittable};

/// A quadrilateral (parallelogram) defined by a corner and two edge vectors
#[derive(Debug, Clone)]
pub struct Quad {
    /// Corner point
    pub q: Vec3,
    /// First edge vector
    pub u: Vec3,
    /// Second edge vector
    pub v: Vec3,
    /// Normal vector
    normal: Vec3,
    /// Plane constant D where Ax + By + Cz = D
    d: f32,
    /// Precomputed value for hit test
    w: Vec3,
    /// Material ID
    pub material_id: usize,
}

impl Quad {
    /// Create a quad from corner and two edge vectors
    pub fn new(q: Vec3, u: Vec3, v: Vec3, material_id: usize) -> Self {
        let n = u.cross(&v);
        let normal = n.normalize();
        let d = normal.dot(&q);
        let w = n / n.dot(&n);

        Self {
            q,
            u,
            v,
            normal,
            d,
            w,
            material_id,
        }
    }

    /// Create a quad from 4 corner vertices (must be coplanar)
    pub fn from_vertices(v0: Vec3, v1: Vec3, v2: Vec3, _v3: Vec3, material_id: usize) -> Self {
        // v0 is the corner, v1-v0 and v2-v0 are edges (v3 is ignored, assumed coplanar)
        let u = v1 - v0;
        let v = v2 - v0;
        Self::new(v0, u, v, material_id)
    }

    /// Check if point (alpha, beta) is inside the unit square
    fn is_interior(alpha: f32, beta: f32, rec: &mut HitRecord) -> bool {
        let unit_interval = 0.0..=1.0;
        if !unit_interval.contains(&alpha) || !unit_interval.contains(&beta) {
            return false;
        }

        rec.uv = (alpha, beta);
        true
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let denom = self.normal.dot(&ray.direction);

        // Ray is parallel to plane
        if denom.abs() < 1e-8 {
            return None;
        }

        // Compute t where ray intersects plane
        let t = (self.d - self.normal.dot(&ray.origin)) / denom;

        // Check bounds
        if t < t_min || t > t_max {
            return None;
        }

        // Compute intersection point
        let intersection = ray.at(t);

        // Check if intersection is within quad
        let planar_hit = intersection - self.q;
        let alpha = self.w.dot(&planar_hit.cross(&self.v));
        let beta = self.w.dot(&self.u.cross(&planar_hit));

        let mut rec = HitRecord {
            t,
            point: intersection,
            normal: Vec3::zero(),
            uv: (0.0, 0.0),
            front_face: false,
            material_id: self.material_id,
        };

        if !Self::is_interior(alpha, beta, &mut rec) {
            return None;
        }

        rec.set_face_normal(ray, self.normal);
        Some(rec)
    }

    fn bounding_box(&self) -> Option<AABB> {
        // Compute bounding box from all four corners
        let p0 = self.q;
        let p1 = self.q + self.u;
        let p2 = self.q + self.v;
        let p3 = self.q + self.u + self.v;

        let min = p0.min(&p1).min(&p2).min(&p3);
        let max = p0.max(&p1).max(&p2).max(&p3);

        // Pad thin boxes
        let epsilon = Vec3::new(0.0001, 0.0001, 0.0001);
        Some(AABB::new(min - epsilon, max + epsilon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_hit() {
        // Unit quad in XY plane at z=0
        let quad = Quad::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );

        // Ray hitting center
        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = quad.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!((rec.t - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quad_miss_outside() {
        let quad = Quad::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );

        // Ray missing (outside quad bounds)
        let ray = Ray::new(Vec3::new(2.0, 0.5, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(quad.hit(&ray, 0.001, f32::INFINITY).is_none());
    }

    #[test]
    fn test_quad_miss_parallel() {
        let quad = Quad::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );

        // Ray parallel to quad
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(quad.hit(&ray, 0.001, f32::INFINITY).is_none());
    }

    #[test]
    fn test_quad_uv() {
        let quad = Quad::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );

        // Ray hitting corner
        let ray = Ray::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = quad.hit(&ray, 0.001, f32::INFINITY).unwrap();
        assert!((hit.uv.0 - 0.0).abs() < 1e-6);
        assert!((hit.uv.1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_quad_bounding_box() {
        let quad = Quad::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            0,
        );

        let bounds = quad.bounding_box().unwrap();
        assert!(bounds.min.x <= 0.0);
        assert!(bounds.max.x >= 1.0);
        assert!(bounds.max.y >= 1.0);
    }

    #[test]
    fn test_quad_from_vertices() {
        let quad = Quad::from_vertices(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            0,
        );

        let ray = Ray::new(Vec3::new(0.5, 0.5, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(quad.hit(&ray, 0.001, f32::INFINITY).is_some());
    }
}
