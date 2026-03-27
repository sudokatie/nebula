//! Scatter record for material sampling

use crate::math::{Vec3, Ray};

/// Result of scattering a ray
#[derive(Clone)]
pub struct ScatterRecord {
    /// Color attenuation
    pub attenuation: Vec3,
    /// Scattered ray
    pub scattered: Ray,
    /// PDF of the sampled direction
    pub pdf: f32,
}

impl ScatterRecord {
    pub fn new(attenuation: Vec3, scattered: Ray) -> Self {
        Self {
            attenuation,
            scattered,
            pdf: 1.0,
        }
    }

    pub fn with_pdf(attenuation: Vec3, scattered: Ray, pdf: f32) -> Self {
        Self {
            attenuation,
            scattered,
            pdf,
        }
    }
}
