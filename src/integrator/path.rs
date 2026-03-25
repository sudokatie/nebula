//! Path tracing integrator

use crate::math::{Vec3, Ray};
use crate::scene::Scene;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Path tracing integrator
pub struct PathIntegrator {
    max_depth: u32,
}

impl PathIntegrator {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth }
    }

    /// Trace a ray through the scene
    pub fn trace(&self, ray: &Ray, scene: &Scene, rng: &mut Xoshiro256PlusPlus) -> Vec3 {
        self.trace_recursive(ray, scene, self.max_depth, rng)
    }

    fn trace_recursive(&self, ray: &Ray, scene: &Scene, depth: u32, rng: &mut Xoshiro256PlusPlus) -> Vec3 {
        if depth == 0 {
            return Vec3::zero();
        }

        // Check for intersection
        if let Some(hit) = scene.hit(ray, 0.001, f32::INFINITY) {
            // Get emission from material
            let emission = scene.material(hit.material_id)
                .map(|m| m.emit())
                .unwrap_or_else(Vec3::zero);

            // Try to scatter
            if let Some(mat) = scene.material(hit.material_id) {
                if let Some(scatter) = mat.scatter(ray, &hit, rng) {
                    let bounced = self.trace_recursive(&scatter.scattered, scene, depth - 1, rng);
                    return emission + scatter.attenuation * bounced;
                }
            }

            return emission;
        }

        // Background color (sky gradient)
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}
