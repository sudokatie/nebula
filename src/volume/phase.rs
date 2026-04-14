//! Phase functions for volumetric scattering.
//!
//! Phase functions describe the angular distribution of scattered light.
//! p(ω_i, ω_o) gives the probability density of scattering from direction
//! ω_i to direction ω_o.

use crate::math::Vec3;
use rand::{Rng, RngCore};
use std::f32::consts::PI;

/// Phase function trait for volume scattering.
pub trait PhaseFunction: Send + Sync {
    /// Evaluate the phase function for the given cosine of the angle between
    /// incoming and outgoing directions.
    fn eval(&self, cos_theta: f32) -> f32;

    /// Sample an outgoing direction given an incoming direction.
    /// Returns (direction, pdf).
    fn sample_dir(&self, wo: &Vec3, rng: &mut dyn RngCore) -> (Vec3, f32);

    /// PDF for sampling the given direction pair.
    fn pdf(&self, cos_theta: f32) -> f32 {
        self.eval(cos_theta)
    }
}

/// Isotropic phase function - uniform scattering in all directions.
#[derive(Debug, Clone, Copy)]
pub struct Isotropic;

impl Isotropic {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Isotropic {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseFunction for Isotropic {
    fn eval(&self, _cos_theta: f32) -> f32 {
        // Uniform over sphere: 1 / (4π)
        1.0 / (4.0 * PI)
    }

    fn sample_dir(&self, _wo: &Vec3, rng: &mut dyn RngCore) -> (Vec3, f32) {
        // Uniform sphere sampling
        let z: f32 = rng.gen_range(-1.0..1.0);
        let r = (1.0_f32 - z * z).max(0.0).sqrt();
        let phi: f32 = rng.gen_range(0.0..(2.0 * PI));
        let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        (dir, 1.0 / (4.0 * PI))
    }
}

/// Henyey-Greenstein phase function.
///
/// The standard phase function for volume scattering. The asymmetry parameter g
/// controls the scattering direction:
/// - g = 0: isotropic scattering
/// - g > 0: forward scattering (light continues mostly forward)
/// - g < 0: backward scattering (light reflects back)
///
/// Typical values:
/// - Smoke/fog: g ≈ 0.0 to 0.3
/// - Clouds: g ≈ 0.85
/// - Tissue: g ≈ 0.9
#[derive(Debug, Clone, Copy)]
pub struct HenyeyGreenstein {
    /// Asymmetry parameter, range [-1, 1].
    pub g: f32,
}

impl HenyeyGreenstein {
    pub fn new(g: f32) -> Self {
        Self { g: g.clamp(-0.999, 0.999) }
    }

    /// Create an isotropic-like HG function (g=0).
    pub fn isotropic() -> Self {
        Self::new(0.0)
    }

    /// Create a forward-scattering phase function.
    pub fn forward(strength: f32) -> Self {
        Self::new(strength.clamp(0.0, 0.999))
    }

    /// Create a backward-scattering phase function.
    pub fn backward(strength: f32) -> Self {
        Self::new(-strength.clamp(0.0, 0.999))
    }
}

impl Default for HenyeyGreenstein {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl PhaseFunction for HenyeyGreenstein {
    fn eval(&self, cos_theta: f32) -> f32 {
        // p(cos_θ) = (1 - g²) / (4π * (1 + g² - 2g*cos_θ)^(3/2))
        let g2 = self.g * self.g;
        let denom = 1.0 + g2 - 2.0 * self.g * cos_theta;
        (1.0 - g2) / (4.0 * PI * denom.powf(1.5))
    }

    fn sample_dir(&self, wo: &Vec3, rng: &mut dyn RngCore) -> (Vec3, f32) {
        // Sample cos_theta from the HG distribution
        let cos_theta: f32 = if self.g.abs() < 1e-3 {
            // Near-isotropic: uniform sampling
            rng.gen_range(-1.0..1.0)
        } else {
            // Inversion method for HG
            let u: f32 = rng.gen();
            let g2 = self.g * self.g;
            let term = (1.0 - g2) / (1.0 - self.g + 2.0 * self.g * u);
            (1.0 + g2 - term * term) / (2.0 * self.g)
        };

        let sin_theta = (1.0_f32 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi: f32 = rng.gen_range(0.0..(2.0 * PI));

        // Build local coordinate system around -wo (incoming direction)
        let (u_axis, v_axis) = coordinate_system(&(-*wo));
        
        // Construct scattered direction
        let dir = u_axis * (sin_theta * phi.cos())
            + v_axis * (sin_theta * phi.sin())
            + (-*wo) * cos_theta;

        let pdf = self.eval(cos_theta);
        (dir.normalize(), pdf)
    }
}

/// Build an orthonormal coordinate system from a single vector.
fn coordinate_system(v: &Vec3) -> (Vec3, Vec3) {
    let w = v.normalize();
    let a = if w.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let u = w.cross(&a).normalize();
    let v = w.cross(&u);
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_isotropic_eval_constant() {
        let iso = Isotropic::new();
        let expected = 1.0 / (4.0 * PI);
        assert!((iso.eval(1.0) - expected).abs() < 1e-6);
        assert!((iso.eval(0.0) - expected).abs() < 1e-6);
        assert!((iso.eval(-1.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_isotropic_sample_unit_vector() {
        let iso = Isotropic::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        
        for _ in 0..100 {
            let (dir, _pdf) = iso.sample_dir(&wo, &mut rng);
            let len = dir.length();
            assert!((len - 1.0).abs() < 1e-5, "direction not normalized: {}", len);
        }
    }

    #[test]
    fn test_hg_isotropic_matches() {
        let hg = HenyeyGreenstein::new(0.0);
        let iso = Isotropic::new();
        
        // For g=0, HG should equal isotropic
        for cos_theta in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let hg_val = hg.eval(cos_theta);
            let iso_val = iso.eval(cos_theta);
            assert!((hg_val - iso_val).abs() < 1e-5,
                "HG(0) != isotropic at cos_theta={}: {} vs {}", cos_theta, hg_val, iso_val);
        }
    }

    #[test]
    fn test_hg_forward_scattering() {
        let hg = HenyeyGreenstein::new(0.8);
        
        // Forward scattering (g > 0) should have higher probability
        // for small angles (cos_theta near 1)
        let forward = hg.eval(1.0);
        let perpendicular = hg.eval(0.0);
        let backward = hg.eval(-1.0);
        
        assert!(forward > perpendicular, "forward should be > perpendicular");
        assert!(perpendicular > backward, "perpendicular should be > backward");
    }

    #[test]
    fn test_hg_backward_scattering() {
        let hg = HenyeyGreenstein::new(-0.8);
        
        // Backward scattering (g < 0) should favor cos_theta near -1
        let forward = hg.eval(1.0);
        let backward = hg.eval(-1.0);
        
        assert!(backward > forward, "backward should dominate for g < 0");
    }

    #[test]
    fn test_hg_normalization() {
        // Integrate HG over sphere should give 1
        let hg = HenyeyGreenstein::new(0.5);
        
        // Numerical integration over cos_theta ∈ [-1, 1]
        // ∫p(cos_θ) * 2π * d(cos_θ) = 1
        let n = 1000;
        let mut sum = 0.0;
        for i in 0..n {
            let cos_theta = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
            sum += hg.eval(cos_theta) * 2.0 * PI * (2.0 / n as f32);
        }
        
        assert!((sum - 1.0).abs() < 0.01, "HG should integrate to 1, got {}", sum);
    }

    #[test]
    fn test_hg_sample_direction_valid() {
        let hg = HenyeyGreenstein::new(0.6);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let wo = Vec3::new(1.0, 0.0, 0.0).normalize();
        
        for _ in 0..100 {
            let (dir, pdf) = hg.sample_dir(&wo, &mut rng);
            assert!((dir.length() - 1.0).abs() < 1e-5, "sampled direction not unit");
            assert!(pdf > 0.0, "pdf should be positive");
        }
    }

    #[test]
    fn test_coordinate_system_orthonormal() {
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();
        let (u, w) = coordinate_system(&v);
        
        assert!((u.length() - 1.0).abs() < 1e-5);
        assert!((w.length() - 1.0).abs() < 1e-5);
        assert!(u.dot(&v).abs() < 1e-5);
        assert!(w.dot(&v).abs() < 1e-5);
        assert!(u.dot(&w).abs() < 1e-5);
    }
}
