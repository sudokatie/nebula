//! Textured Lambertian material

use std::sync::Arc;

use crate::math::{Vec3, Ray};
use crate::geometry::HitRecord;
use crate::texture::Texture;
use super::{Material, ScatterRecord};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Lambertian material with texture support
pub struct TexturedLambertian {
    pub texture: Arc<dyn Texture>,
}

impl TexturedLambertian {
    pub fn new(texture: Arc<dyn Texture>) -> Self {
        Self { texture }
    }
}

impl Material for TexturedLambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord, rng: &mut Xoshiro256PlusPlus) -> Option<ScatterRecord> {
        // Use cosine-weighted hemisphere sampling
        let (dir, _pdf) = crate::integrator::sample_cosine_hemisphere(&hit.normal, rng);
        
        let scatter_direction = if dir.near_zero() {
            hit.normal
        } else {
            dir
        };

        // Sample texture at hit point
        let albedo = self.texture.value(hit.uv.0, hit.uv.1, &hit.point);
        
        Some(ScatterRecord {
            attenuation: albedo,
            scattered: Ray::new(hit.point, scatter_direction),
        })
    }

    fn eval(&self, _ray_in: &Ray, hit: &HitRecord, _dir_out: &Vec3) -> Vec3 {
        // Lambertian BSDF with texture
        let albedo = self.texture.value(hit.uv.0, hit.uv.1, &hit.point);
        albedo / std::f32::consts::PI
    }

    fn pdf(&self, _ray_in: &Ray, hit: &HitRecord, dir_out: &Vec3) -> f32 {
        let cos_theta = hit.normal.dot(dir_out).max(0.0);
        cos_theta / std::f32::consts::PI
    }

    fn is_specular(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::SolidColor;
    use rand::SeedableRng;

    #[test]
    fn test_textured_lambertian() {
        let tex = Arc::new(SolidColor::rgb(1.0, 0.0, 0.0));
        let mat = TexturedLambertian::new(tex);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.5, 0.5),
            front_face: true,
            material_id: 0,
        };
        
        let scatter = mat.scatter(&ray, &hit, &mut rng);
        assert!(scatter.is_some());
        
        // Check attenuation is red
        let s = scatter.unwrap();
        assert!((s.attenuation.x - 1.0).abs() < 1e-6);
        assert!(s.attenuation.y.abs() < 1e-6);
    }

    #[test]
    fn test_textured_eval() {
        let tex = Arc::new(SolidColor::rgb(0.8, 0.2, 0.1));
        let mat = TexturedLambertian::new(tex);
        
        let ray = Ray::new(Vec3::zero(), Vec3::new(0.0, -1.0, 0.0));
        let hit = HitRecord {
            t: 1.0,
            point: Vec3::zero(),
            normal: Vec3::new(0.0, 1.0, 0.0),
            uv: (0.0, 0.0),
            front_face: true,
            material_id: 0,
        };
        let dir_out = Vec3::new(0.0, 1.0, 0.0);
        
        let bsdf = mat.eval(&ray, &hit, &dir_out);
        // Should be albedo / pi
        assert!((bsdf.x - 0.8 / std::f32::consts::PI).abs() < 1e-6);
    }
}
