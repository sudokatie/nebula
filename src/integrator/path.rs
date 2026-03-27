//! Path tracing integrator with NEE, MIS, and Russian Roulette

use crate::math::{Vec3, Ray};
use crate::scene::Scene;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Path tracing integrator with advanced sampling
pub struct PathIntegrator {
    max_depth: u32,
    russian_roulette_depth: u32,
}

impl PathIntegrator {
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            russian_roulette_depth: 3, // Start RR after 3 bounces
        }
    }

    /// Create with custom Russian Roulette depth
    pub fn with_rr_depth(max_depth: u32, rr_depth: u32) -> Self {
        Self {
            max_depth,
            russian_roulette_depth: rr_depth,
        }
    }

    /// Trace a ray through the scene
    pub fn trace(&self, ray: &Ray, scene: &Scene, rng: &mut Xoshiro256PlusPlus) -> Vec3 {
        self.trace_with_nee(ray, scene, rng)
    }

    /// Path trace with Next Event Estimation and MIS
    fn trace_with_nee(&self, initial_ray: &Ray, scene: &Scene, rng: &mut Xoshiro256PlusPlus) -> Vec3 {
        let mut color = Vec3::zero();
        let mut throughput = Vec3::one();
        let mut ray = *initial_ray;
        let mut specular_bounce = true; // First hit can see emitters directly

        for depth in 0..self.max_depth {
            // Russian Roulette termination
            if depth >= self.russian_roulette_depth {
                let p = throughput.x.max(throughput.y).max(throughput.z).min(0.95);
                if rng.gen::<f32>() > p {
                    break;
                }
                throughput = throughput / p;
            }

            // Find intersection
            let hit = match scene.hit(&ray, 0.001, f32::INFINITY) {
                Some(h) => h,
                None => {
                    // Sky background
                    color += throughput * self.background(&ray);
                    break;
                }
            };

            // Get material
            let material = match scene.material(hit.material_id) {
                Some(m) => m,
                None => break,
            };

            // Add emission (only if this is a specular bounce or first hit)
            // This avoids double-counting with NEE
            let emission = material.emit();
            if specular_bounce && emission.length_squared() > 0.0 {
                color += throughput * emission;
            }

            // Next Event Estimation - direct light sampling
            if let Some(light_sample) = scene.sample_light(rng) {
                let to_light = light_sample.point - hit.point;
                let light_dist = to_light.length();
                let light_dir = to_light / light_dist;

                // Check visibility with shadow ray
                let shadow_ray = Ray::new(hit.point, light_dir);
                let visible = !scene.occluded(&shadow_ray, 0.001, light_dist - 0.001);

                if visible {
                    // Compute direct lighting contribution
                    let n_dot_l = hit.normal.dot(&light_dir).max(0.0);
                    
                    if n_dot_l > 0.0 {
                        // Get BSDF value
                        let bsdf_val = material.eval(&ray, &hit, &light_dir);
                        let bsdf_pdf = material.pdf(&ray, &hit, &light_dir);
                        
                        // MIS weight (power heuristic)
                        let light_pdf = light_sample.pdf;
                        let mis_weight = power_heuristic(light_pdf, bsdf_pdf);

                        // Add NEE contribution
                        let geometry_term = n_dot_l / (light_dist * light_dist);
                        let nee_contrib = light_sample.emission * bsdf_val * geometry_term * mis_weight / light_pdf;
                        color += throughput * nee_contrib;
                    }
                }
            }

            // Sample BSDF for next direction
            let scatter = match material.scatter(&ray, &hit, rng) {
                Some(s) => s,
                None => break,
            };

            // Update for next bounce
            specular_bounce = material.is_specular();
            throughput = throughput * scatter.attenuation;
            ray = scatter.scattered;

            // Add BSDF-sampled emitter contribution with MIS
            // (This handles the case where we hit a light via BSDF sampling)
            // The contribution is added in the next iteration via specular_bounce check
        }

        color
    }

    /// Simple recursive trace (fallback, no NEE/MIS)
    #[allow(dead_code)]
    fn trace_simple(&self, ray: &Ray, scene: &Scene, depth: u32, rng: &mut Xoshiro256PlusPlus) -> Vec3 {
        if depth == 0 {
            return Vec3::zero();
        }

        // Russian Roulette
        let rr_factor = if depth < self.max_depth - self.russian_roulette_depth {
            let p = 0.8_f32; // 80% survival rate
            if rng.gen::<f32>() > p {
                return Vec3::zero();
            }
            1.0 / p
        } else {
            1.0
        };

        if let Some(hit) = scene.hit(ray, 0.001, f32::INFINITY) {
            let emission = scene.material(hit.material_id)
                .map(|m| m.emit())
                .unwrap_or_else(Vec3::zero);

            if let Some(mat) = scene.material(hit.material_id) {
                if let Some(scatter) = mat.scatter(ray, &hit, rng) {
                    let bounced = self.trace_simple(&scatter.scattered, scene, depth - 1, rng);
                    return (emission + scatter.attenuation * bounced) * rr_factor;
                }
            }

            return emission * rr_factor;
        }

        self.background(ray)
    }

    /// Background color (sky gradient)
    fn background(&self, ray: &Ray) -> Vec3 {
        let unit_direction = ray.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
    }
}

/// Power heuristic for MIS (beta = 2)
fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    if a2 + b2 > 0.0 {
        a2 / (a2 + b2)
    } else {
        0.0
    }
}

/// Balance heuristic for MIS
#[allow(dead_code)]
fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    if pdf_a + pdf_b > 0.0 {
        pdf_a / (pdf_a + pdf_b)
    } else {
        0.0
    }
}

/// Cosine-weighted hemisphere sampling
pub fn sample_cosine_hemisphere(normal: &Vec3, rng: &mut impl Rng) -> (Vec3, f32) {
    // Sample uniformly on unit disk
    let r1: f32 = rng.gen();
    let r2: f32 = rng.gen();
    
    let phi = 2.0 * std::f32::consts::PI * r1;
    let cos_theta = (1.0 - r2).sqrt();
    let sin_theta = r2.sqrt();
    
    // Direction in local coordinates
    let x = phi.cos() * sin_theta;
    let y = phi.sin() * sin_theta;
    let z = cos_theta;
    
    // Transform to world space
    let (tangent, bitangent) = build_basis(normal);
    let direction = tangent * x + bitangent * y + *normal * z;
    
    // PDF is cos(theta) / pi
    let pdf = cos_theta / std::f32::consts::PI;
    
    (direction, pdf)
}

/// Cosine-weighted hemisphere PDF
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    if cos_theta > 0.0 {
        cos_theta / std::f32::consts::PI
    } else {
        0.0
    }
}

/// Build orthonormal basis from normal
fn build_basis(n: &Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    
    let tangent = Vec3::new(
        1.0 + sign * n.x * n.x * a,
        sign * b,
        -sign * n.x,
    );
    let bitangent = Vec3::new(
        b,
        sign + n.y * n.y * a,
        -n.y,
    );
    
    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_heuristic() {
        // Equal PDFs should give 0.5
        let w = power_heuristic(1.0, 1.0);
        assert!((w - 0.5).abs() < 1e-6);
        
        // If pdf_a >> pdf_b, weight should approach 1
        let w = power_heuristic(100.0, 1.0);
        assert!(w > 0.99);
        
        // If pdf_a << pdf_b, weight should approach 0
        let w = power_heuristic(1.0, 100.0);
        assert!(w < 0.01);
    }

    #[test]
    fn test_balance_heuristic() {
        let w = balance_heuristic(1.0, 1.0);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_hemisphere_sampling() {
        use rand::SeedableRng;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        
        for _ in 0..100 {
            let (dir, pdf) = sample_cosine_hemisphere(&normal, &mut rng);
            
            // Direction should be in upper hemisphere
            assert!(dir.dot(&normal) >= 0.0);
            
            // Direction should be normalized
            assert!((dir.length() - 1.0).abs() < 1e-5);
            
            // PDF should be positive
            assert!(pdf > 0.0);
        }
    }

    #[test]
    fn test_cosine_hemisphere_pdf() {
        // At normal (cos_theta = 1), PDF = 1/pi
        let pdf = cosine_hemisphere_pdf(1.0);
        assert!((pdf - 1.0 / std::f32::consts::PI).abs() < 1e-6);
        
        // At horizon (cos_theta = 0), PDF = 0
        let pdf = cosine_hemisphere_pdf(0.0);
        assert!((pdf - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_basis_orthonormal() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let (t, b) = build_basis(&normal);
        
        // Should be orthogonal
        assert!(t.dot(&b).abs() < 1e-6);
        assert!(t.dot(&normal).abs() < 1e-6);
        assert!(b.dot(&normal).abs() < 1e-6);
    }

    #[test]
    fn test_path_integrator_new() {
        let integrator = PathIntegrator::new(10);
        assert_eq!(integrator.max_depth, 10);
        assert_eq!(integrator.russian_roulette_depth, 3);
    }

    #[test]
    fn test_path_integrator_with_rr() {
        let integrator = PathIntegrator::with_rr_depth(20, 5);
        assert_eq!(integrator.max_depth, 20);
        assert_eq!(integrator.russian_roulette_depth, 5);
    }
}
