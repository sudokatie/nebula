//! CPU renderer with tiled rendering and optional SIMD ray packets

use crate::math::Vec3;
use crate::camera::Camera;
use crate::scene::Scene;
use crate::integrator::PathIntegrator;
use crate::sampler::Sampler;
use rayon::prelude::*;

/// Tile size for cache-efficient rendering
const TILE_SIZE: u32 = 16;

/// CPU-based path tracer with tiled rendering
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

    /// Render scene (multi-threaded with tiled rendering)
    pub fn render_parallel(&self, scene: &Scene, camera: &Camera, _threads: usize) -> Vec<Vec3> {
        self.render_tiled(scene, camera)
    }

    /// Tiled rendering for better cache efficiency
    fn render_tiled(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let width = self.width;
        let height = self.height;
        let samples = self.samples_per_pixel;

        // Calculate tiles
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let total_tiles = tiles_x * tiles_y;

        // Generate tile indices
        let tile_indices: Vec<(u32, u32)> = (0..total_tiles)
            .map(|i| (i % tiles_x, i / tiles_x))
            .collect();

        // Render tiles in parallel
        let tile_results: Vec<(u32, u32, Vec<Vec3>)> = tile_indices
            .par_iter()
            .map(|&(tile_x, tile_y)| {
                let start_x = tile_x * TILE_SIZE;
                let start_y = tile_y * TILE_SIZE;
                let end_x = (start_x + TILE_SIZE).min(width);
                let end_y = (start_y + TILE_SIZE).min(height);

                let tile_width = end_x - start_x;
                let tile_height = end_y - start_y;
                let mut tile_pixels = Vec::with_capacity((tile_width * tile_height) as usize);

                let seed = (tile_y * tiles_x + tile_x) as u64 * 12345;
                let mut sampler = Sampler::new(seed);

                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let mut color = Vec3::zero();

                        for _ in 0..samples {
                            let u = (x as f32 + sampler.random()) / (width - 1) as f32;
                            let v = ((height - 1 - y) as f32 + sampler.random()) / (height - 1) as f32;
                            
                            let ray = camera.get_ray(u, v, sampler.inner_mut());
                            color += self.integrator.trace(&ray, scene, sampler.inner_mut());
                        }

                        tile_pixels.push(color / samples as f32);
                    }
                }

                (tile_x, tile_y, tile_pixels)
            })
            .collect();

        // Assemble final image from tiles
        let mut pixels = vec![Vec3::zero(); (width * height) as usize];

        for (tile_x, tile_y, tile_pixels) in tile_results {
            let start_x = tile_x * TILE_SIZE;
            let start_y = tile_y * TILE_SIZE;
            let end_x = (start_x + TILE_SIZE).min(width);
            let end_y = (start_y + TILE_SIZE).min(height);
            let tile_width = end_x - start_x;

            for (i, color) in tile_pixels.into_iter().enumerate() {
                let local_x = i as u32 % tile_width;
                let local_y = i as u32 / tile_width;
                let global_x = start_x + local_x;
                let global_y = start_y + local_y;
                pixels[(global_y * width + global_x) as usize] = color;
            }
        }

        pixels
    }

    /// Render with progressive updates (for preview)
    pub fn render_progressive<F>(&self, scene: &Scene, camera: &Camera, mut callback: F)
    where
        F: FnMut(&[Vec3], u32), // pixels, sample count
    {
        let width = self.width;
        let height = self.height;
        let mut accumulator = vec![Vec3::zero(); (width * height) as usize];
        let mut sampler = Sampler::new(42);

        for sample in 1..=self.samples_per_pixel {
            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + sampler.random()) / (width - 1) as f32;
                    let v = ((height - 1 - y) as f32 + sampler.random()) / (height - 1) as f32;
                    
                    let ray = camera.get_ray(u, v, sampler.inner_mut());
                    let color = self.integrator.trace(&ray, scene, sampler.inner_mut());
                    
                    let idx = (y * width + x) as usize;
                    accumulator[idx] += color;
                }
            }

            // Call callback with current averaged result
            let averaged: Vec<Vec3> = accumulator
                .iter()
                .map(|&c| c / sample as f32)
                .collect();
            callback(&averaged, sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;
    use crate::material::Lambertian;

    fn create_test_scene() -> (Scene, Camera) {
        let mut scene = Scene::new();
        
        let mat_id = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat_id));
        scene.build_bvh();

        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            90.0,
            1.0,
            0.0,
            1.0,
        );

        (scene, camera)
    }

    #[test]
    fn test_cpu_renderer_new() {
        let renderer = CpuRenderer::new(100, 100, 10, 5);
        assert_eq!(renderer.width, 100);
        assert_eq!(renderer.height, 100);
        assert_eq!(renderer.samples_per_pixel, 10);
    }

    #[test]
    fn test_cpu_render_single_threaded() {
        let (scene, camera) = create_test_scene();
        let renderer = CpuRenderer::new(10, 10, 1, 2);
        
        let pixels = renderer.render(&scene, &camera);
        assert_eq!(pixels.len(), 100);
    }

    #[test]
    fn test_cpu_render_tiled() {
        let (scene, camera) = create_test_scene();
        let renderer = CpuRenderer::new(32, 32, 1, 2);
        
        let pixels = renderer.render_tiled(&scene, &camera);
        assert_eq!(pixels.len(), 32 * 32);
    }

    #[test]
    fn test_cpu_render_parallel() {
        let (scene, camera) = create_test_scene();
        let renderer = CpuRenderer::new(32, 32, 2, 2);
        
        let pixels = renderer.render_parallel(&scene, &camera, 4);
        assert_eq!(pixels.len(), 32 * 32);
    }

    #[test]
    fn test_tile_size() {
        assert_eq!(TILE_SIZE, 16);
    }

    #[test]
    fn test_render_progressive() {
        let (scene, camera) = create_test_scene();
        let renderer = CpuRenderer::new(8, 8, 3, 2);
        
        let mut call_count = 0;
        renderer.render_progressive(&scene, &camera, |pixels, sample| {
            assert_eq!(pixels.len(), 64);
            assert!(sample >= 1 && sample <= 3);
            call_count += 1;
        });
        
        assert_eq!(call_count, 3);
    }
}
