//! Scene representation

mod loader;
mod obj_loader;

pub use loader::{load_scene, RenderSettings};
pub use obj_loader::load_obj;

use rand::Rng;

use crate::math::{Vec3, Ray};
use crate::geometry::{HitRecord, Hittable, Sphere, Triangle, Quad, Mesh};
use crate::accel::BVH;
use crate::material::Material;

/// Light sample for NEE
pub struct LightSample {
    /// Point on light surface
    pub point: Vec3,
    /// Normal at sample point
    pub normal: Vec3,
    /// Emission from light
    pub emission: Vec3,
    /// PDF of sampling this point
    pub pdf: f32,
}

/// A scene with objects and materials
pub struct Scene {
    bvh: Option<BVH>,
    primitives: Vec<Box<dyn Hittable>>,
    materials: Vec<Box<dyn Material>>,
    /// Indices of emissive objects (for light sampling)
    light_indices: Vec<usize>,
    /// Cached light areas for sampling
    light_areas: Vec<f32>,
    total_light_area: f32,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            bvh: None,
            primitives: Vec::new(),
            materials: Vec::new(),
            light_indices: Vec::new(),
            light_areas: Vec::new(),
            total_light_area: 0.0,
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

    /// Add a quad to the scene
    pub fn add_quad(&mut self, quad: Quad) {
        self.primitives.push(Box::new(quad));
        self.bvh = None;
    }

    /// Add a mesh to the scene
    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.primitives.push(Box::new(mesh));
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
        // Track light sources
        self.light_indices.clear();
        self.light_areas.clear();
        self.total_light_area = 0.0;

        // Note: We need to track lights before moving primitives
        // For now, we'll rebuild light indices after BVH construction
        // This is a simplification - proper implementation would track
        // primitive -> material mapping

        let primitives = std::mem::take(&mut self.primitives);
        self.bvh = Some(BVH::new(primitives));
    }

    /// Register a light source (call after adding primitives)
    pub fn register_light(&mut self, primitive_idx: usize, area: f32) {
        self.light_indices.push(primitive_idx);
        self.light_areas.push(area);
        self.total_light_area += area;
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

    /// Check if ray is occluded (for shadow rays)
    pub fn occluded(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        self.hit(ray, t_min, t_max).is_some()
    }

    /// Sample a light source for NEE
    pub fn sample_light(&self, rng: &mut impl Rng) -> Option<LightSample> {
        if self.light_indices.is_empty() || self.total_light_area <= 0.0 {
            return None;
        }

        // Sample light proportional to area
        let target = rng.gen::<f32>() * self.total_light_area;
        let mut cumulative = 0.0;
        let mut selected_idx = 0;

        for (i, &area) in self.light_areas.iter().enumerate() {
            cumulative += area;
            if target <= cumulative {
                selected_idx = i;
                break;
            }
        }

        // Get the selected light's primitive index
        let prim_idx = self.light_indices[selected_idx];
        let area = self.light_areas[selected_idx];

        // Sample a point on the primitive
        // For now, return a simple sample (proper implementation would sample based on primitive type)
        // This is a placeholder - actual implementation needs access to primitive geometry

        // Get material emission
        // Note: This is simplified - proper implementation would look up the material
        // For now, return white emission
        Some(LightSample {
            point: Vec3::zero(), // Would be sampled from primitive
            normal: Vec3::new(0.0, -1.0, 0.0),
            emission: Vec3::one() * 10.0, // Default emission
            pdf: 1.0 / area,
        })
    }

    /// Get number of light sources
    pub fn light_count(&self) -> usize {
        self.light_indices.len()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
