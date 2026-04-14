//! Homogeneous participating medium - constant density throughout.
//!
//! Simple but efficient model for uniform fog, smoke, or atmosphere.
//! Uses closed-form transmittance and analytic sampling.

use super::{PhaseFunction, Transmittance, Volume, VolumeSample, HenyeyGreenstein};
use crate::math::{Ray, Vec3};
use rand::Rng;

/// Homogeneous volume with constant density.
#[derive(Debug, Clone)]
pub struct HomogeneousVolume {
    /// Extinction coefficient (absorption + scattering).
    sigma_t: Vec3,
    /// Scattering coefficient.
    sigma_s: Vec3,
    /// Phase function for scattering direction.
    phase: HenyeyGreenstein,
    /// Cached max extinction for sampling.
    max_sigma_t: f32,
}

impl HomogeneousVolume {
    /// Create a new homogeneous volume.
    ///
    /// # Arguments
    /// * `sigma_t` - Extinction coefficient (higher = denser)
    /// * `albedo` - Single-scattering albedo [0, 1] (ratio of scattering to extinction)
    /// * `g` - Asymmetry parameter for Henyey-Greenstein phase function
    pub fn new(sigma_t: Vec3, albedo: Vec3, g: f32) -> Self {
        let sigma_s = Vec3::new(
            sigma_t.x * albedo.x,
            sigma_t.y * albedo.y,
            sigma_t.z * albedo.z,
        );
        let max = sigma_t.x.max(sigma_t.y).max(sigma_t.z);
        
        Self {
            sigma_t,
            sigma_s,
            phase: HenyeyGreenstein::new(g),
            max_sigma_t: max,
        }
    }

    /// Create uniform gray fog.
    pub fn fog(density: f32, albedo: f32, g: f32) -> Self {
        Self::new(
            Vec3::new(density, density, density),
            Vec3::new(albedo, albedo, albedo),
            g,
        )
    }

    /// Create a thin fog layer.
    pub fn thin_fog() -> Self {
        Self::fog(0.05, 0.9, 0.0)
    }

    /// Create dense smoke.
    pub fn smoke() -> Self {
        Self::fog(0.3, 0.6, 0.3)
    }

    /// Create atmospheric haze.
    pub fn haze() -> Self {
        Self::fog(0.01, 0.95, 0.8)
    }

    /// Sample distance using analytic method (Beer's law).
    fn sample_distance(&self, rng: &mut impl Rng) -> f32 {
        // For homogeneous media, sample t from exponential distribution
        // t = -ln(1 - ξ) / σ_t = -ln(ξ) / σ_t
        let u: f32 = rng.gen();
        -u.ln() / self.max_sigma_t
    }
}

impl Volume for HomogeneousVolume {
    fn sigma_t(&self, _point: &Vec3) -> Vec3 {
        self.sigma_t
    }

    fn sigma_s(&self, _point: &Vec3) -> Vec3 {
        self.sigma_s
    }

    fn max_sigma_t(&self) -> f32 {
        self.max_sigma_t
    }

    fn sample(&self, ray: &Ray, t_max: f32, rng: &mut impl Rng) -> VolumeSample {
        let t = self.sample_distance(rng);
        
        if t < t_max {
            // Scattering event inside the volume
            let position = ray.at(t);
            let transmittance = Transmittance::from_extinction(self.sigma_t, t);
            
            VolumeSample {
                t,
                scattered: true,
                transmittance: transmittance.value,
                position,
            }
        } else {
            // Ray exits the volume without scattering
            let transmittance = Transmittance::from_extinction(self.sigma_t, t_max);
            
            VolumeSample {
                t: t_max,
                scattered: false,
                transmittance: transmittance.value,
                position: ray.at(t_max),
            }
        }
    }

    fn transmittance(&self, _ray: &Ray, t_max: f32, _rng: &mut impl Rng) -> Transmittance {
        // Closed-form for homogeneous media: T = exp(-σ_t * t)
        Transmittance::from_extinction(self.sigma_t, t_max)
    }

    fn phase(&self) -> &dyn PhaseFunction {
        &self.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_homogeneous_creation() {
        let vol = HomogeneousVolume::new(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        let point = Vec3::zero();
        assert!((vol.sigma_t(&point).x - 0.5).abs() < 1e-6);
        assert!((vol.sigma_s(&point).x - 0.45).abs() < 1e-6); // 0.5 * 0.9
        assert!((vol.max_sigma_t() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fog_presets() {
        let thin = HomogeneousVolume::thin_fog();
        let smoke = HomogeneousVolume::smoke();
        let haze = HomogeneousVolume::haze();
        
        // Smoke should be denser than fog
        assert!(smoke.max_sigma_t() > thin.max_sigma_t());
        // Haze should be thinnest
        assert!(haze.max_sigma_t() < thin.max_sigma_t());
    }

    #[test]
    fn test_transmittance_decreases_with_distance() {
        let vol = HomogeneousVolume::fog(0.5, 0.9, 0.0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let t1 = vol.transmittance(&ray, 1.0, &mut rng);
        let t2 = vol.transmittance(&ray, 2.0, &mut rng);
        let t3 = vol.transmittance(&ray, 4.0, &mut rng);
        
        assert!(t1.value.x > t2.value.x, "transmittance should decrease");
        assert!(t2.value.x > t3.value.x, "transmittance should decrease");
    }

    #[test]
    fn test_transmittance_zero_distance() {
        let vol = HomogeneousVolume::smoke();
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let t = vol.transmittance(&ray, 0.0, &mut rng);
        assert!((t.value.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_produces_valid_results() {
        let vol = HomogeneousVolume::fog(0.3, 0.9, 0.0);
        let ray = Ray::new(Vec3::zero(), Vec3::new(1.0, 0.0, 0.0));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let mut scattered_count = 0;
        let mut exit_count = 0;
        
        for _ in 0..1000 {
            let sample = vol.sample(&ray, 10.0, &mut rng);
            
            assert!(sample.t >= 0.0);
            assert!(sample.t <= 10.0 || !sample.scattered);
            assert!(sample.transmittance.x >= 0.0 && sample.transmittance.x <= 1.0);
            
            if sample.scattered {
                scattered_count += 1;
            } else {
                exit_count += 1;
            }
        }
        
        // Should have some of both (statistical)
        assert!(scattered_count > 0, "should have some scattering events");
        assert!(exit_count > 0, "should have some exits");
    }

    #[test]
    fn test_sigma_constant_throughout() {
        let vol = HomogeneousVolume::smoke();
        
        let p1 = Vec3::new(0.0, 0.0, 0.0);
        let p2 = Vec3::new(100.0, 50.0, -30.0);
        let p3 = Vec3::new(-1000.0, 0.0, 1000.0);
        
        assert!((vol.sigma_t(&p1).x - vol.sigma_t(&p2).x).abs() < 1e-6);
        assert!((vol.sigma_t(&p2).y - vol.sigma_t(&p3).y).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_calculation() {
        let vol = HomogeneousVolume::new(
            Vec3::new(1.0, 2.0, 4.0),
            Vec3::new(0.5, 0.25, 0.75),
            0.0,
        );
        
        let point = Vec3::zero();
        let albedo = vol.albedo(&point);
        
        assert!((albedo.x - 0.5).abs() < 1e-6);
        assert!((albedo.y - 0.25).abs() < 1e-6);
        assert!((albedo.z - 0.75).abs() < 1e-6);
    }
}
