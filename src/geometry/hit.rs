//! Hit record and hittable trait

use crate::math::{Vec3, Ray};
use crate::accel::AABB;

/// Record of a ray-surface intersection
#[derive(Debug, Clone)]
pub struct HitRecord {
    /// Distance along ray
    pub t: f32,
    /// Point of intersection
    pub point: Vec3,
    /// Surface normal (always points against ray)
    pub normal: Vec3,
    /// UV coordinates
    pub uv: (f32, f32),
    /// Whether ray hit front face
    pub front_face: bool,
    /// Material index
    pub material_id: usize,
}

impl HitRecord {
    /// Set normal direction based on ray direction
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

/// Trait for objects that can be hit by rays
pub trait Hittable: Send + Sync {
    /// Test ray intersection in [t_min, t_max]
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    
    /// Get bounding box (None for infinite objects)
    fn bounding_box(&self) -> Option<AABB>;
}
