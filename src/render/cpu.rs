//! CPU renderer with optional parallelism

use crate::math::Vec3;
use crate::camera::Camera;
use crate::scene::Scene;
use crate::integrator::PathIntegrator;
use crate::sampler::Sampler;
use rayon::prelude::*;

/// CPU-based path tracer
pub struct CpuRenderer {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    integrator: PathIntegrator,
}

impl CpuRenderer {
    pub fn new(width: u32, height: u32, samples: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            samples_per_pixel: samples,
            integrator: PathIntegrator::new(depth),
        }
    }

    /// Render scene (single-threaded, for debugging)
    pub fn render(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let mut pixels = vec![Vec3::zero(); (self.width * self.height) as usize];
        let mut sampler = Sampler::new(42);

        for y in 0..self.height {
            for x in 0..self.width {
                let mut color = Vec3::zero();
                
                for _ in 0..self.samples_per_pixel {
                    let u = (x as f32 + sampler.random()) / (self.width - 1) as f32;
                    let v = ((self.height - 1 - y) as f32 + sampler.random()) / (self.height - 1) as f32;
                    
                    let ray = camera.get_ray(u, v, sampler.inner_mut());
                    color += self.integrator.trace(&ray, scene, sampler.inner_mut());
                }
                
                pixels[(y * self.width + x) as usize] = color / self.samples_per_pixel as f32;
            }
        }

        pixels
    }

    /// Render scene (multi-threaded)
    pub fn render_parallel(&self, scene: &Scene, camera: &Camera, _threads: usize) -> Vec<Vec3> {
        let width = self.width;
        let height = self.height;
        let samples = self.samples_per_pixel;

        // Create pixel indices
        let indices: Vec<(u32, u32)> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .collect();

        // Render in parallel
        let pixels: Vec<Vec3> = indices
            .par_iter()
            .map(|&(x, y)| {
                let seed = (y * width + x) as u64;
                let mut sampler = Sampler::new(seed);
                let mut color = Vec3::zero();

                for _ in 0..samples {
                    let u = (x as f32 + sampler.random()) / (width - 1) as f32;
                    let v = ((height - 1 - y) as f32 + sampler.random()) / (height - 1) as f32;
                    
                    let ray = camera.get_ray(u, v, sampler.inner_mut());
                    color += self.integrator.trace(&ray, scene, sampler.inner_mut());
                }

                color / samples as f32
            })
            .collect();

        pixels
    }
}
