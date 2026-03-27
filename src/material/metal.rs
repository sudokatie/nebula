//! Metal (specular) material with GGX microfacets

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use super::{Material, ScatterRecord};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Metal reflective material with GGX microfacet distribution
pub struct Metal {
    pub albedo: Vec3,
    pub roughness: f32,
}

impl Metal {
    pub fn new(albedo: Vec3, roughness: f32) -> Self {
        Self {
            albedo,
            roughness: roughness.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        let reflected = ray.direction.normalize().reflect(&hit.normal);

        // Apply GGX microfacet perturbation for roughness > 0
        let scattered_dir = if self.roughness > 0.001 {
            // Sample GGX distribution
            let h = sample_ggx_vndf(&hit.normal, &(-ray.direction), self.roughness, rng);
            let scattered = ray.direction.normalize().reflect(&h);
            
            // Fallback to simple perturbation if GGX gives bad direction
            if scattered.dot(&hit.normal) <= 0.0 {
                reflected + Vec3::random_in_unit_sphere(rng) * self.roughness
            } else {
                scattered
            }
        } else {
            reflected
        };

        let scattered = Ray::new(hit.point, scattered_dir);
        
        // Only scatter if reflected ray is on same side as normal
        if scattered.direction.dot(&hit.normal) > 0.0 {
            Some(ScatterRecord {
                attenuation: self.albedo,
                scattered,
            })
        } else {
            None
        }
    }

    fn eval(&self, ray_in: &Ray, hit: &HitRecord, dir_out: &Vec3) -> Vec3 {
        // For perfect specular (roughness = 0), this is a delta distribution
        if self.roughness < 0.001 {
            return self.albedo;
        }

        // GGX microfacet BRDF
        let v = (-ray_in.direction).normalize();
        let l = dir_out.normalize();
        let h = (v + l).normalize();
        
        let n_dot_v = hit.normal.dot(&v).max(0.001);
        let n_dot_l = hit.normal.dot(&l).max(0.0);
        let n_dot_h = hit.normal.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.0);

        if n_dot_l <= 0.0 {
            return Vec3::zero();
        }

        let d = ggx_distribution(n_dot_h, self.roughness);
        let g = smith_g(n_dot_v, n_dot_l, self.roughness);
        let f = fresnel_schlick(v_dot_h, self.albedo);

        // Cook-Torrance BRDF
        f * (d * g / (4.0 * n_dot_v * n_dot_l))
    }

    fn pdf(&self, ray_in: &Ray, hit: &HitRecord, dir_out: &Vec3) -> f32 {
        if self.roughness < 0.001 {
            // Delta distribution
            return 1.0;
        }

        let v = (-ray_in.direction).normalize();
        let l = dir_out.normalize();
        let h = (v + l).normalize();
        
        let n_dot_h = hit.normal.dot(&h).max(0.0);
        let v_dot_h = v.dot(&h).max(0.001);

        // PDF = D(h) * (n·h) / (4 * v·h)
        let d = ggx_distribution(n_dot_h, self.roughness);
        d * n_dot_h / (4.0 * v_dot_h)
    }

    fn is_specular(&self) -> bool {
        // Only truly specular if roughness is ~0
        self.roughness < 0.001
    }
}

/// Sample GGX/Trowbridge-Reitz distribution (VNDF - Visible Normal Distribution Function)
fn sample_ggx_vndf(normal: &Vec3, view: &Vec3, roughness: f32, rng: &mut impl Rng) -> Vec3 {
    let alpha = roughness * roughness;
    
    // Build orthonormal basis
    let (tangent, bitangent) = build_orthonormal_basis(normal);
    
    // Transform view to local space
    let v_local = Vec3::new(
        view.dot(&tangent),
        view.dot(&bitangent),
        view.dot(normal),
    );
    
    // Sample visible microfacet normal (simplified spherical cap sampling)
    let u1: f32 = rng.gen();
    let u2: f32 = rng.gen();
    
    // GGX importance sampling
    let phi = 2.0 * std::f32::consts::PI * u1;
    let cos_theta_sq = (1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2);
    let cos_theta = cos_theta_sq.sqrt();
    let sin_theta = (1.0 - cos_theta_sq).max(0.0).sqrt();
    
    // Microfacet normal in local space
    let h_local = Vec3::new(
        sin_theta * phi.cos(),
        sin_theta * phi.sin(),
        cos_theta,
    );
    
    // Transform back to world space
    tangent * h_local.x + bitangent * h_local.y + *normal * h_local.z
}

/// Build orthonormal basis from a normal vector
fn build_orthonormal_basis(n: &Vec3) -> (Vec3, Vec3) {
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

/// GGX normal distribution function D(h)
#[allow(dead_code)]
pub fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_sq = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    alpha_sq / (std::f32::consts::PI * denom * denom)
}

/// Smith's geometric shadowing function G1
#[allow(dead_code)]
pub fn smith_g1(n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_sq = alpha * alpha;
    let n_dot_v_sq = n_dot_v * n_dot_v;
    2.0 * n_dot_v / (n_dot_v + (alpha_sq + (1.0 - alpha_sq) * n_dot_v_sq).sqrt())
}

/// Combined Smith geometric shadowing G
#[allow(dead_code)]
pub fn smith_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    smith_g1(n_dot_v, roughness) * smith_g1(n_dot_l, roughness)
}

/// Fresnel-Schlick approximation
#[allow(dead_code)]
pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::one() - f0) * (1.0 - cos_theta).powi(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggx_distribution() {
        // At n_dot_h = 1 (perfect specular), D should be highest
        let d1 = ggx_distribution(1.0, 0.1);
        let d2 = ggx_distribution(0.5, 0.1);
        assert!(d1 > d2);
    }

    #[test]
    fn test_ggx_distribution_roughness() {
        // Higher roughness should give lower peak
        let d_smooth = ggx_distribution(1.0, 0.1);
        let d_rough = ggx_distribution(1.0, 0.5);
        assert!(d_smooth > d_rough);
    }

    #[test]
    fn test_smith_g1_range() {
        let g = smith_g1(0.5, 0.3);
        assert!(g >= 0.0 && g <= 1.0);
    }

    #[test]
    fn test_fresnel_schlick() {
        let f0 = Vec3::new(0.04, 0.04, 0.04); // Dielectric F0
        
        // At normal incidence, should be close to F0
        let f_normal = fresnel_schlick(1.0, f0);
        assert!((f_normal.x - f0.x).abs() < 0.01);
        
        // At grazing angle, should approach 1
        let f_grazing = fresnel_schlick(0.0, f0);
        assert!(f_grazing.x > 0.9);
    }

    #[test]
    fn test_orthonormal_basis() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let (t, b) = build_orthonormal_basis(&n);
        
        // Should be orthogonal
        assert!(t.dot(&b).abs() < 1e-6);
        assert!(t.dot(&n).abs() < 1e-6);
        assert!(b.dot(&n).abs() < 1e-6);
        
        // Should be unit vectors
        assert!((t.length() - 1.0).abs() < 1e-6);
        assert!((b.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metal_scatter() {
        use rand::SeedableRng;
        let metal = Metal::new(Vec3::new(0.8, 0.6, 0.2), 0.3);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let ray = Ray::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let scatter = metal.scatter(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
        
        // Scattered ray should go up (reflected)
        let s = scatter.unwrap();
        assert!(s.scattered.direction.y > 0.0);
    }

    #[test]
    fn test_perfect_mirror() {
        use rand::SeedableRng;
        let metal = Metal::new(Vec3::one(), 0.0); // Perfect mirror
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let ray = Ray::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        
        let scatter = metal.scatter(&ray, &hit, &mut rng).unwrap();
        // Perfect mirror should reflect exactly
        assert!((scatter.scattered.direction.y - 1.0).abs() < 1e-6);
    }
}
