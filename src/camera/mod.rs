//! Camera model

use crate::math::{Vec3, Ray};
use rand::Rng;

/// Thin lens camera with depth of field
pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    #[allow(dead_code)]
    w: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(
        look_from: Vec3,
        look_at: Vec3,
        up: Vec3,
        vfov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = up.cross(&w).normalize();
        let v = w.cross(&u);

        let origin = look_from;
        let horizontal = u * viewport_width * focus_dist;
        let vertical = v * viewport_height * focus_dist;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w * focus_dist;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius: aperture / 2.0,
        }
    }

    /// Generate ray for normalized coordinates (s, t) in [0,1]
    pub fn get_ray<R: Rng>(&self, s: f32, t: f32, rng: &mut R) -> Ray {
        let rd = Vec3::random_in_unit_disk(rng) * self.lens_radius;
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.origin + offset,
            self.lower_left_corner + self.horizontal * s + self.vertical * t - self.origin - offset,
        )
    }
}
