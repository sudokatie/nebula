//! Sphere primitive

use crate::math::{Vec3, Ray};
use crate::accel::AABB;
use super::{HitRecord, Hittable};

/// A sphere primitive
#[derive(Debug, Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material_id: usize,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32, material_id: usize) -> Self {
        Self { center, radius, material_id }
    }

    /// Get UV coordinates for a point on the sphere
    fn get_uv(point: &Vec3) -> (f32, f32) {
        // Point is on unit sphere centered at origin
        let theta = (-point.y).acos();
        let phi = (-point.z).atan2(point.x) + std::f32::consts::PI;
        
        let u = phi / (2.0 * std::f32::consts::PI);
        let v = theta / std::f32::consts::PI;
        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        
        let sqrtd = discriminant.sqrt();
        
        // Find nearest root in acceptable range
        let mut root = (-half_b - sqrtd) / a;
        if root <= t_min || root >= t_max {
            root = (-half_b + sqrtd) / a;
            if root <= t_min || root >= t_max {
                return None;
            }
        }
        
        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let uv = Self::get_uv(&outward_normal);
        
        let mut rec = HitRecord {
            t: root,
            point,
            geometric_normal: Vec3::zero(),
            normal: Vec3::zero(),
            uv,
            front_face: false,
            material_id: self.material_id,
        };
        // For spheres, geometric and shading normals are the same
        rec.set_normals(ray, outward_normal, outward_normal);
        
        Some(rec)
    }

    fn bounding_box(&self) -> Option<AABB> {
        let radius_vec = Vec3::new(self.radius, self.radius, self.radius);
        Some(AABB::new(
            self.center - radius_vec,
            self.center + radius_vec,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_hit() {
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, -1.0));
        
        let hit = sphere.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        
        let rec = hit.unwrap();
        assert!((rec.t - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_miss() {
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, 0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        
        let hit = sphere.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_none());
    }

    #[test]
    fn test_sphere_inside() {
        let sphere = Sphere::new(Vec3::zero(), 2.0, 0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, 0.0, 1.0));
        
        let hit = sphere.hit(&ray, 0.001, f32::INFINITY);
        assert!(hit.is_some());
        assert!(!hit.unwrap().front_face);
    }
}
