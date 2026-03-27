//! Importance sampling utilities

use crate::math::Vec3;
use std::f32::consts::PI;

/// Cosine-weighted hemisphere sampling (for diffuse)
/// Returns direction in local space where Z is up
pub fn cosine_weighted_hemisphere(u1: f32, u2: f32) -> Vec3 {
    let r = u1.sqrt();
    let theta = 2.0 * PI * u2;
    let x = r * theta.cos();
    let y = r * theta.sin();
    let z = (1.0 - u1).sqrt();
    Vec3::new(x, y, z)
}

/// PDF for cosine-weighted hemisphere sampling
pub fn cosine_weighted_pdf(cos_theta: f32) -> f32 {
    cos_theta.max(0.0) / PI
}

/// Uniform hemisphere sampling
pub fn uniform_hemisphere(u1: f32, u2: f32) -> Vec3 {
    let z = u1;
    let r = (1.0 - z * z).sqrt();
    let phi = 2.0 * PI * u2;
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// PDF for uniform hemisphere sampling
pub fn uniform_hemisphere_pdf() -> f32 {
    1.0 / (2.0 * PI)
}

/// Uniform sphere sampling
pub fn uniform_sphere(u1: f32, u2: f32) -> Vec3 {
    let z = 1.0 - 2.0 * u1;
    let r = (1.0 - z * z).sqrt();
    let phi = 2.0 * PI * u2;
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

/// GGX/Trowbridge-Reitz normal distribution
pub fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// GGX importance sampling - sample microfacet normal
pub fn ggx_sample(u1: f32, u2: f32, roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let a2 = a * a;
    
    let phi = 2.0 * PI * u1;
    let cos_theta = ((1.0 - u2) / (u2 * (a2 - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    
    Vec3::new(
        sin_theta * phi.cos(),
        sin_theta * phi.sin(),
        cos_theta,
    )
}

/// PDF for GGX sampling
pub fn ggx_pdf(n_dot_h: f32, roughness: f32) -> f32 {
    ggx_d(n_dot_h, roughness) * n_dot_h
}

/// Schlick's Fresnel approximation
pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0).powi(5);
    f0 + (Vec3::new(1.0, 1.0, 1.0) - f0) * t
}

/// Single-value Fresnel for dielectrics
pub fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        return 1.0; // Total internal reflection
    }
    
    let cos_t = (1.0 - sin2_t).sqrt();
    let r_parallel = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    
    (r_parallel * r_parallel + r_perp * r_perp) / 2.0
}

/// Build orthonormal basis from normal
pub fn build_onb(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let w = n.normalize();
    let a = if w.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let v = w.cross(&a).normalize();
    let u = w.cross(&v);
    (u, v, w)
}

/// Transform local direction to world space using ONB
pub fn local_to_world(local: Vec3, u: Vec3, v: Vec3, w: Vec3) -> Vec3 {
    u * local.x + v * local.y + w * local.z
}

/// Power heuristic for MIS (beta = 2)
pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    a2 / (a2 + b2)
}

/// Balance heuristic for MIS
pub fn balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    pdf_a / (pdf_a + pdf_b)
}

/// Sample point on sphere surface
pub fn sample_sphere_surface(center: Vec3, radius: f32, u1: f32, u2: f32) -> (Vec3, Vec3) {
    let dir = uniform_sphere(u1, u2);
    let point = center + dir * radius;
    (point, dir)
}

/// PDF for sphere sampling (uniform)
pub fn sphere_pdf(radius: f32) -> f32 {
    1.0 / (4.0 * PI * radius * radius)
}

/// Sample direction toward sphere from point
pub fn sample_sphere_solid_angle(
    center: Vec3,
    radius: f32,
    from: Vec3,
    u1: f32,
    u2: f32,
) -> Option<(Vec3, f32)> {
    let to_center = center - from;
    let dist_sq = to_center.length_squared();
    let dist = dist_sq.sqrt();
    
    if dist < radius {
        return None; // Inside sphere
    }
    
    let sin_theta_max_sq = radius * radius / dist_sq;
    let cos_theta_max = (1.0 - sin_theta_max_sq).sqrt();
    
    // Sample cone
    let cos_theta = 1.0 + u1 * (cos_theta_max - 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = 2.0 * PI * u2;
    
    // Build basis
    let w = to_center / dist;
    let (u, v, w) = build_onb(w);
    
    let dir = local_to_world(
        Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta),
        u, v, w
    );
    
    // PDF is 1 / solid angle
    let pdf = 1.0 / (2.0 * PI * (1.0 - cos_theta_max));
    
    Some((dir, pdf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_weighted_hemisphere() {
        let dir = cosine_weighted_hemisphere(0.5, 0.5);
        assert!(dir.z >= 0.0);
        assert!((dir.length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_weighted_pdf() {
        let pdf = cosine_weighted_pdf(1.0);
        assert!((pdf - 1.0 / PI).abs() < 0.001);
    }

    #[test]
    fn test_ggx_sample() {
        let h = ggx_sample(0.5, 0.5, 0.5);
        assert!(h.z >= 0.0);
        assert!((h.length() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fresnel_schlick() {
        let f0 = Vec3::new(0.04, 0.04, 0.04);
        let f = fresnel_schlick(1.0, f0);
        assert!((f.x - 0.04).abs() < 0.01);
        
        let f_grazing = fresnel_schlick(0.0, f0);
        assert!(f_grazing.x > 0.9);
    }

    #[test]
    fn test_build_onb() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let (u, v, w) = build_onb(n);
        assert!(u.dot(&v).abs() < 0.001);
        assert!(u.dot(&w).abs() < 0.001);
        assert!(v.dot(&w).abs() < 0.001);
    }

    #[test]
    fn test_power_heuristic() {
        let w = power_heuristic(1.0, 1.0);
        assert!((w - 0.5).abs() < 0.01);
        
        let w2 = power_heuristic(2.0, 1.0);
        assert!(w2 > 0.5);
    }
}
