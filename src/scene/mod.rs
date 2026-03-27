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

/// Light source info for sampling
struct LightInfo {
    /// Type of light primitive
    light_type: LightType,
    /// Material ID for emission lookup
    material_id: usize,
    /// Area of the light
    area: f32,
}

#[derive(Clone)]
enum LightType {
    Sphere { center: Vec3, radius: f32 },
    Quad { q: Vec3, u: Vec3, v: Vec3, normal: Vec3 },
}

/// A scene with objects and materials
pub struct Scene {
    bvh: Option<BVH>,
    primitives: Vec<Box<dyn Hittable>>,
    materials: Vec<Box<dyn Material>>,
    /// Light source info for sampling
    lights: Vec<LightInfo>,
    /// Total light area for PDF
    total_light_area: f32,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            bvh: None,
            primitives: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            total_light_area: 0.0,
        }
    }

    /// Add a sphere to the scene
    pub fn add_sphere(&mut self, sphere: Sphere) {
        let material_id = sphere.material_id;
        let center = sphere.center;
        let radius = sphere.radius;
        self.primitives.push(Box::new(sphere));
        self.bvh = None;
        
        // Track for potential light registration after materials are set
        self.maybe_register_sphere_light(center, radius, material_id);
    }

    fn maybe_register_sphere_light(&mut self, center: Vec3, radius: f32, material_id: usize) {
        // Check if material is emissive (if material exists)
        if let Some(mat) = self.materials.get(material_id) {
            if mat.is_emissive() {
                let area = 4.0 * std::f32::consts::PI * radius * radius;
                self.lights.push(LightInfo {
                    light_type: LightType::Sphere { center, radius },
                    material_id,
                    area,
                });
                self.total_light_area += area;
            }
        }
    }

    /// Add a triangle to the scene
    pub fn add_triangle(&mut self, triangle: Triangle) {
        self.primitives.push(Box::new(triangle));
        self.bvh = None;
    }

    /// Add a quad to the scene (potentially emissive)
    pub fn add_quad(&mut self, quad: Quad) {
        let material_id = quad.material_id;
        let q = quad.q;
        let u = quad.u;
        let v = quad.v;
        let normal = u.cross(&v).normalize();
        self.primitives.push(Box::new(quad));
        self.bvh = None;
        
        // Check if emissive
        if let Some(mat) = self.materials.get(material_id) {
            if mat.is_emissive() {
                let area = u.cross(&v).length();
                self.lights.push(LightInfo {
                    light_type: LightType::Quad { q, u, v, normal },
                    material_id,
                    area,
                });
                self.total_light_area += area;
            }
        }
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

    /// Check if ray is occluded (for shadow rays)
    pub fn occluded(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        self.hit(ray, t_min, t_max).is_some()
    }

    /// Sample a light source for NEE
    pub fn sample_light(&self, rng: &mut impl Rng) -> Option<LightSample> {
        if self.lights.is_empty() || self.total_light_area <= 0.0 {
            return None;
        }

        // Sample light proportional to area
        let target = rng.gen::<f32>() * self.total_light_area;
        let mut cumulative = 0.0;
        let mut selected_idx = 0;

        for (i, light) in self.lights.iter().enumerate() {
            cumulative += light.area;
            if target <= cumulative {
                selected_idx = i;
                break;
            }
        }

        let light = &self.lights[selected_idx];
        
        // Sample point on the light based on its type
        let (point, normal) = match &light.light_type {
            LightType::Sphere { center, radius } => {
                // Sample uniform point on sphere surface
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let z = 1.0 - 2.0 * u1;
                let r = (1.0 - z * z).sqrt();
                let phi = 2.0 * std::f32::consts::PI * u2;
                let normal = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                let point = *center + normal * *radius;
                (point, normal)
            }
            LightType::Quad { q, u, v, normal } => {
                // Sample uniform point on quad
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let point = *q + *u * u1 + *v * u2;
                (point, *normal)
            }
        };

        // Get emission from material
        let emission = self.materials.get(light.material_id)
            .map(|m| m.emit())
            .unwrap_or_else(Vec3::zero);

        // PDF is 1/total_area (uniform sampling across all lights)
        let pdf = 1.0 / self.total_light_area;

        Some(LightSample {
            point,
            normal,
            emission,
            pdf,
        })
    }

    /// Get number of light sources
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Check if scene has any lights
    pub fn has_lights(&self) -> bool {
        !self.lights.is_empty()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
