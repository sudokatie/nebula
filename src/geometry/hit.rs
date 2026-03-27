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
    /// Geometric normal (from raw geometry, for shadow terminator)
    pub geometric_normal: Vec3,
    /// Shading normal (interpolated/from normal map, for shading)
    pub normal: Vec3,
    /// UV coordinates
    pub uv: (f32, f32),
    /// Whether ray hit front face
    pub front_face: bool,
    /// Material index
    pub material_id: usize,
}

impl HitRecord {
    /// Create a new hit record
    pub fn new(
        t: f32,
        point: Vec3,
        geometric_normal: Vec3,
        shading_normal: Vec3,
        uv: (f32, f32),
        material_id: usize,
    ) -> Self {
        Self {
            t,
            point,
            geometric_normal,
            normal: shading_normal,
            uv,
            front_face: true,
            material_id,
        }
    }

    /// Set normal direction based on ray direction
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.geometric_normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
        // Also flip shading normal if not already set differently
        if self.normal.near_zero() {
            self.normal = self.geometric_normal;
        } else if !self.front_face {
            self.normal = -self.normal;
        }
    }

    /// Set both geometric and shading normals
    pub fn set_normals(&mut self, ray: &Ray, geometric: Vec3, shading: Vec3) {
        self.front_face = ray.direction.dot(&geometric) < 0.0;
        if self.front_face {
            self.geometric_normal = geometric;
            self.normal = shading;
        } else {
            self.geometric_normal = -geometric;
            self.normal = -shading;
        }
    }
}

impl Default for HitRecord {
    fn default() -> Self {
        Self {
            t: 0.0,
            point: Vec3::zero(),
            geometric_normal: Vec3::zero(),
            normal: Vec3::zero(),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        }
    }
}

/// Trait for objects that can be hit by rays
pub trait Hittable: Send + Sync {
    /// Test ray intersection in [t_min, t_max]
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    
    /// Get bounding box (None for infinite objects)
    fn bounding_box(&self) -> Option<AABB>;
}
