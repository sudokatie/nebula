//! Path tracing integrator with NEE and MIS

use crate::math::{Vec3, Ray, RayDifferential};
use crate::scene::Scene;
use crate::sampler::sampling::power_heuristic;
use rand::Rng;

/// Path tracing integrator with Russian Roulette, NEE, and MIS
pub struct PathIntegrator {
    max_depth: u32,
    min_depth_for_rr: u32,
    use_nee: bool,
    use_mis: bool,
    use_differentials: bool,
}

impl PathIntegrator {
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            min_depth_for_rr: 3,
            use_nee: true,
            use_mis: true,
            use_differentials: true,
        }
    }

    pub fn simple(max_depth: u32) -> Self {
        Self {
            max_depth,
            min_depth_for_rr: 3,
            use_nee: false,
            use_mis: false,
            use_differentials: false,
        }
    }

    pub fn with_nee(mut self, enabled: bool) -> Self {
        self.use_nee = enabled;
        self
    }

    pub fn with_mis(mut self, enabled: bool) -> Self {
        self.use_mis = enabled;
        self
    }

    pub fn with_differentials(mut self, enabled: bool) -> Self {
        self.use_differentials = enabled;
        self
    }

    /// Trace a ray through the scene (basic Ray version for backward compatibility)
    pub fn trace<R: Rng>(&self, ray: &Ray, scene: &Scene, rng: &mut R) -> Vec3 {
        if self.use_differentials {
            let ray_diff = RayDifferential::from(*ray);
            self.trace_differential(&ray_diff, scene, rng)
        } else {
            self.trace_internal(ray, scene, rng)
        }
    }

    /// Trace a ray differential through the scene (for texture filtering)
    pub fn trace_differential<R: Rng>(&self, ray: &RayDifferential, scene: &Scene, rng: &mut R) -> Vec3 {
        let mut color = Vec3::zero();
        let mut throughput = Vec3::new(1.0, 1.0, 1.0);
        let mut current_ray = ray.ray;
        let mut current_diff = *ray;
        let mut depth = 0;
        let mut specular_bounce = true;
        let mut prev_bsdf_pdf = 1.0f32;

        loop {
            if depth >= self.max_depth {
                break;
            }

            // Intersect scene
            let hit = match scene.hit(&current_ray, 0.001, f32::INFINITY) {
                Some(h) => h,
                None => {
                    // Background
                    color = color + throughput * self.background(&current_ray);
                    break;
                }
            };

            let mat = match scene.material(hit.material_id) {
                Some(m) => m,
                None => break,
            };

            // Compute differential footprint for texture filtering (mipmap selection)
            let footprint = if current_diff.has_differentials {
                let (du, dv) = current_diff.compute_differentials_at(hit.point, hit.normal);
                // Use average of dU and dV as the filter footprint
                (du + dv) * 0.5
            } else {
                0.0 // No differentials - use base mip level
            };

            // Add emission
            let emission = mat.emit();
            if emission.length_squared() > 0.0 {
                if specular_bounce || !self.use_nee {
                    // First hit or specular bounce: add full emission
                    color = color + throughput * emission;
                } else if self.use_mis {
                    // BSDF hit on a light: apply MIS weight
                    // Light PDF for this hit
                    let light_pdf = self.light_pdf_for_hit(scene, &hit, &current_ray);
                    if light_pdf > 0.0 {
                        let weight = power_heuristic(prev_bsdf_pdf, light_pdf);
                        color = color + throughput * emission * weight;
                    }
                }
                // For emissive surfaces that don't scatter, we're done
                if mat.scatter_dyn(&current_ray, &hit, rng).is_none() {
                    break;
                }
            }

            // Next Event Estimation (direct light sampling)
            if self.use_nee && !mat.is_delta() {
                let wo = (-current_ray.direction).normalize();
                let direct = self.sample_lights(scene, &hit, &wo, rng);
                color = color + throughput * direct;
            }

            // Russian Roulette
            if depth >= self.min_depth_for_rr {
                let p_continue = throughput.max_component().min(0.95);
                if rng.gen::<f32>() > p_continue {
                    break;
                }
                throughput = throughput / p_continue;
            }

            // Sample BSDF for next direction (with texture LOD)
            let scatter = match mat.scatter_with_lod(&current_ray, &hit, rng, footprint) {
                Some(s) => s,
                None => break,
            };

            specular_bounce = mat.is_delta();
            prev_bsdf_pdf = scatter.pdf;
            
            // Update throughput
            // For importance-sampled BSDF: throughput *= (BSDF * cos_theta) / PDF
            // scatter.attenuation already contains BSDF * cos_theta for most materials
            throughput = throughput * scatter.attenuation;
            
            // Apply PDF for non-delta materials
            if !mat.is_delta() && scatter.pdf > 0.0 {
                throughput = throughput / scatter.pdf;
            }

            // Update ray differential for next bounce
            if current_diff.has_differentials {
                // Scale differentials based on surface roughness
                let scale_factor = if mat.is_delta() { 1.0 } else { 1.0 + mat.roughness() };
                current_diff.scale_differentials(scale_factor);
                current_diff.ray = scatter.scattered;
            }

            current_ray = scatter.scattered;
            depth += 1;
        }

        color
    }

    /// Internal trace without differentials
    fn trace_internal<R: Rng>(&self, ray: &Ray, scene: &Scene, rng: &mut R) -> Vec3 {
        let mut color = Vec3::zero();
        let mut throughput = Vec3::new(1.0, 1.0, 1.0);
        let mut current_ray = *ray;
        let mut depth = 0;
        let mut specular_bounce = true;
        let mut prev_bsdf_pdf = 1.0f32;

        loop {
            if depth >= self.max_depth {
                break;
            }

            // Intersect scene
            let hit = match scene.hit(&current_ray, 0.001, f32::INFINITY) {
                Some(h) => h,
                None => {
                    // Background
                    color = color + throughput * self.background(&current_ray);
                    break;
                }
            };

            let mat = match scene.material(hit.material_id) {
                Some(m) => m,
                None => break,
            };

            // Add emission
            let emission = mat.emit();
            if emission.length_squared() > 0.0 {
                if specular_bounce || !self.use_nee {
                    color = color + throughput * emission;
                } else if self.use_mis {
                    let light_pdf = self.light_pdf_for_hit(scene, &hit, &current_ray);
                    if light_pdf > 0.0 {
                        let weight = power_heuristic(prev_bsdf_pdf, light_pdf);
                        color = color + throughput * emission * weight;
                    }
                }
                if mat.scatter_dyn(&current_ray, &hit, rng).is_none() {
                    break;
                }
            }

            // Next Event Estimation
            if self.use_nee && !mat.is_delta() {
                let wo = (-current_ray.direction).normalize();
                let direct = self.sample_lights(scene, &hit, &wo, rng);
                color = color + throughput * direct;
            }

            // Russian Roulette
            if depth >= self.min_depth_for_rr {
                let p_continue = throughput.max_component().min(0.95);
                if rng.gen::<f32>() > p_continue {
                    break;
                }
                throughput = throughput / p_continue;
            }

            // Sample BSDF
            let scatter = match mat.scatter_dyn(&current_ray, &hit, rng) {
                Some(s) => s,
                None => break,
            };

            specular_bounce = mat.is_delta();
            prev_bsdf_pdf = scatter.pdf;
            throughput = throughput * scatter.attenuation;
            
            if !mat.is_delta() && scatter.pdf > 0.0 {
                throughput = throughput / scatter.pdf;
            }

            current_ray = scatter.scattered;
            depth += 1;
        }

        color
    }

    /// Compute the light PDF for hitting a specific point (for MIS)
    fn light_pdf_for_hit(&self, scene: &Scene, hit: &crate::geometry::HitRecord, ray: &Ray) -> f32 {
        let lights = scene.emissive_objects();
        if lights.is_empty() {
            return 0.0;
        }
        
        // Find if this hit is on a light
        for light in lights {
            if light.material_id == hit.material_id {
                // Compute PDF for hitting this light
                let to_light = hit.point - ray.origin;
                let dist_sq = to_light.length_squared();
                let light_n_dot_l = (-ray.direction.normalize()).dot(&hit.normal).abs();
                
                if light_n_dot_l > 0.0 {
                    // PDF = distance^2 / (cos_theta * area) * (1 / num_lights)
                    return dist_sq / (light_n_dot_l * light.area() * lights.len() as f32);
                }
            }
        }
        
        0.0
    }

    /// Sample direct lighting from emissive objects
    fn sample_lights<R: Rng>(
        &self,
        scene: &Scene,
        hit: &crate::geometry::HitRecord,
        wo: &Vec3,
        rng: &mut R,
    ) -> Vec3 {
        let lights = scene.emissive_objects();
        if lights.is_empty() {
            return Vec3::zero();
        }

        let mut direct = Vec3::zero();

        // Sample one light randomly
        let light_idx = rng.gen_range(0..lights.len());
        let light = &lights[light_idx];
        let num_lights = lights.len() as f32;

        // Sample point on light
        let (light_pos, light_normal, light_emission) = 
            scene.sample_light(light, rng.gen(), rng.gen());
        
        let to_light = light_pos - hit.point;
        let dist_sq = to_light.length_squared();
        let dist = dist_sq.sqrt();
        let light_dir = to_light / dist;

        // Check visibility with shadow ray
        let shadow_ray = Ray::new(hit.point, light_dir);
        if let Some(shadow_hit) = scene.hit(&shadow_ray, 0.001, dist - 0.001) {
            // Blocked by something other than the light
            if shadow_hit.material_id != light.material_id {
                return Vec3::zero();
            }
        }

        // Compute contribution
        let n_dot_l = hit.normal.dot(&light_dir).max(0.0);
        let light_n_dot_l = (-light_dir).dot(&light_normal).max(0.0);

        if n_dot_l > 0.0 && light_n_dot_l > 0.0 {
            // Get material BSDF value
            let mat = scene.material(hit.material_id).unwrap();
            let bsdf = mat.eval(&light_dir, wo, &hit.normal);

            // Light PDF (area to solid angle conversion)
            // PDF = distance^2 / (cos_theta_light * area) * (1 / num_lights)
            let light_pdf = dist_sq / (light_n_dot_l * light.area()) * num_lights;

            if self.use_mis {
                // BSDF PDF for this direction
                let bsdf_pdf = mat.pdf(&light_dir, wo, &hit.normal);
                let weight = power_heuristic(light_pdf, bsdf_pdf);
                // L = BSDF * Le * cos_theta * weight / light_pdf
                direct = bsdf * light_emission * n_dot_l * weight / light_pdf.max(0.0001);
            } else {
                direct = bsdf * light_emission * n_dot_l / light_pdf.max(0.0001);
            }
        }

        direct
    }

    /// Background color (sky gradient)
    fn background(&self, ray: &Ray) -> Vec3 {
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_integrator_new() {
        let integrator = PathIntegrator::new(50);
        assert_eq!(integrator.max_depth, 50);
        assert!(integrator.use_nee);
        assert!(integrator.use_mis);
        assert!(integrator.use_differentials);
    }

    #[test]
    fn test_path_integrator_simple() {
        let integrator = PathIntegrator::simple(50);
        assert!(!integrator.use_nee);
        assert!(!integrator.use_mis);
        assert!(!integrator.use_differentials);
    }

    #[test]
    fn test_path_integrator_config() {
        let integrator = PathIntegrator::new(50)
            .with_nee(false)
            .with_mis(false)
            .with_differentials(false);
        assert!(!integrator.use_nee);
        assert!(!integrator.use_mis);
        assert!(!integrator.use_differentials);
    }
}
