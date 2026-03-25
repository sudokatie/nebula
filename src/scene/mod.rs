//! Scene representation

mod loader;

pub use loader::{load_scene, RenderSettings};

use crate::math::Ray;
use crate::geometry::{HitRecord, Hittable, Sphere, Triangle};
use crate::accel::BVH;
use crate::material::Material;

/// A scene with objects and materials
pub struct Scene {
    bvh: Option<BVH>,
    primitives: Vec<Box<dyn Hittable>>,
    materials: Vec<Box<dyn Material>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            bvh: None,
            primitives: Vec::new(),
            materials: Vec::new(),
        }
    }

    /// Add a sphere to the scene
    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.primitives.push(Box::new(sphere));
        self.bvh = None; // Invalidate BVH
    }

    /// Add a triangle to the scene
    pub fn add_triangle(&mut self, triangle: Triangle) {
        self.primitives.push(Box::new(triangle));
        self.bvh = None;
    }

    /// Add a material and return its ID
    pub fn add_material(&mut self, material: Box<dyn Material>) -> usize {
        let id = self.materials.len();
        self.materials.push(material);
        id
    }

    /// Get material by ID
    pub fn material(&self, id: usize) -> Option<&dyn Material> {
        self.materials.get(id).map(|m| m.as_ref())
    }

    /// Build the BVH (must call before rendering)
    pub fn build_bvh(&mut self) {
        let primitives = std::mem::take(&mut self.primitives);
        self.bvh = Some(BVH::new(primitives));
    }

    /// Test ray intersection
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if let Some(bvh) = &self.bvh {
            bvh.hit(ray, t_min, t_max)
        } else {
            // Linear scan fallback
            let mut closest: Option<HitRecord> = None;
            let mut closest_t = t_max;

            for primitive in &self.primitives {
                if let Some(rec) = primitive.hit(ray, t_min, closest_t) {
                    closest_t = rec.t;
                    closest = Some(rec);
                }
            }
            closest
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
