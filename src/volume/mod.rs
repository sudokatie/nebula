//! Volumetric rendering - participating media for fog, smoke, subsurface scattering.
//!
//! Implements homogeneous and heterogeneous volumes with phase function sampling
//! for physically-based light transport through participating media.

mod homogeneous;
mod heterogeneous;
mod phase;

pub use homogeneous::HomogeneousVolume;
pub use heterogeneous::HeterogeneousVolume;
pub use phase::{PhaseFunction, HenyeyGreenstein, Isotropic};

use crate::math::{Ray, Vec3};

/// Result of sampling a distance through a volume.
#[derive(Debug, Clone)]
pub struct VolumeSample {
    /// Distance traveled through the medium.
    pub t: f32,
    /// Whether a scattering event occurred (vs exiting the medium).
    pub scattered: bool,
    /// Transmittance along the sampled path.
    pub transmittance: Vec3,
    /// Position of the interaction (if scattered).
    pub position: Vec3,
}

/// Result of evaluating transmittance along a ray segment.
#[derive(Debug, Clone, Copy)]
pub struct Transmittance {
    /// Fraction of light transmitted (per channel).
    pub value: Vec3,
}

impl Transmittance {
    pub fn one() -> Self {
        Self { value: Vec3::one() }
    }

    pub fn from_extinction(sigma_t: Vec3, distance: f32) -> Self {
        Self {
            value: Vec3::new(
                (-sigma_t.x * distance).exp(),
                (-sigma_t.y * distance).exp(),
                (-sigma_t.z * distance).exp(),
            ),
        }
    }
}

/// Trait for participating media (volumes).
pub trait Volume: Send + Sync {
    /// Extinction coefficient at a point (absorption + scattering).
    fn sigma_t(&self, point: &Vec3) -> Vec3;

    /// Scattering coefficient at a point.
    fn sigma_s(&self, point: &Vec3) -> Vec3;

    /// Absorption coefficient at a point. Default: sigma_t - sigma_s
    fn sigma_a(&self, point: &Vec3) -> Vec3 {
        let st = self.sigma_t(point);
        let ss = self.sigma_s(point);
        Vec3::new(st.x - ss.x, st.y - ss.y, st.z - ss.z)
    }

    /// Single-scattering albedo at a point. Ratio of scattering to extinction.
    fn albedo(&self, point: &Vec3) -> Vec3 {
        let st = self.sigma_t(point);
        let ss = self.sigma_s(point);
        Vec3::new(
            if st.x > 0.0 { ss.x / st.x } else { 0.0 },
            if st.y > 0.0 { ss.y / st.y } else { 0.0 },
            if st.z > 0.0 { ss.z / st.z } else { 0.0 },
        )
    }

    /// Maximum extinction coefficient in the volume (for delta tracking).
    fn max_sigma_t(&self) -> f32;

    /// Sample a distance through the volume using delta tracking.
    /// Returns the distance traveled and whether scattering occurred.
    fn sample(&self, ray: &Ray, t_max: f32, rng: &mut impl rand::Rng) -> VolumeSample;

    /// Evaluate transmittance along a ray segment.
    fn transmittance(&self, ray: &Ray, t_max: f32, rng: &mut impl rand::Rng) -> Transmittance;

    /// Phase function for this volume.
    fn phase(&self) -> &dyn PhaseFunction;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transmittance_one() {
        let t = Transmittance::one();
        assert!((t.value.x - 1.0).abs() < 1e-6);
        assert!((t.value.y - 1.0).abs() < 1e-6);
        assert!((t.value.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transmittance_from_extinction() {
        let sigma_t = Vec3::new(0.5, 0.5, 0.5);
        let t = Transmittance::from_extinction(sigma_t, 2.0);
        // exp(-0.5 * 2) = exp(-1) ≈ 0.368
        let expected = (-1.0_f32).exp();
        assert!((t.value.x - expected).abs() < 1e-5);
    }

    #[test]
    fn test_transmittance_zero_distance() {
        let sigma_t = Vec3::new(1.0, 1.0, 1.0);
        let t = Transmittance::from_extinction(sigma_t, 0.0);
        assert!((t.value.x - 1.0).abs() < 1e-6);
    }
}
