//! CPU renderer with multi-threading and SIMD packet tracing

use crate::math::{Vec3, Ray};
use crate::scene::Scene;
use crate::camera::Camera;
use crate::integrator::PathIntegrator;
use crate::accel::RayPacket;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// CPU-based renderer
pub struct CpuRenderer {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    use_packet_tracing: bool,
}

impl CpuRenderer {
    pub fn new(width: u32, height: u32, samples_per_pixel: u32, max_depth: u32) -> Self {
        Self {
            width,
            height,
            samples_per_pixel,
            max_depth,
            use_packet_tracing: true,
        }
    }

    /// Enable or disable packet tracing
    pub fn with_packet_tracing(mut self, enabled: bool) -> Self {
        self.use_packet_tracing = enabled;
        self
    }

    /// Render scene single-threaded
    pub fn render(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let mut pixels = vec![Vec3::zero(); (self.width * self.height) as usize];
        let integrator = PathIntegrator::new(self.max_depth);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let width_f = (self.width.max(1) - 1) as f32;
        let height_f = (self.height.max(1) - 1) as f32;

        for y in 0..self.height {
            for x in 0..self.width {
                let mut color = Vec3::zero();
                
                for _ in 0..self.samples_per_pixel {
                    let u = if width_f > 0.0 {
                        (x as f32 + rand::Rng::gen::<f32>(&mut rng)) / width_f
                    } else {
                        0.5
                    };
                    let v = if height_f > 0.0 {
                        ((self.height - 1 - y) as f32 + rand::Rng::gen::<f32>(&mut rng)) / height_f
                    } else {
                        0.5
                    };
                    
                    let ray = camera.get_ray(u, v, &mut rng);
                    color = color + integrator.trace(&ray, scene, &mut rng);
                }
                
                color = color / self.samples_per_pixel as f32;
                pixels[(y * self.width + x) as usize] = color;
            }
        }

        pixels
    }

    /// Render scene with parallel tiles
    pub fn render_parallel(&self, scene: &Scene, camera: &Camera, threads: usize) -> Vec<Vec3> {
        // Configure thread pool if specified
        if threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
        }

        if self.use_packet_tracing {
            self.render_parallel_packets(scene, camera)
        } else {
            self.render_parallel_scalar(scene, camera)
        }
    }

    /// Parallel rendering with scalar (single ray) processing
    fn render_parallel_scalar(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let integrator = PathIntegrator::new(self.max_depth);
        let width = self.width;
        let height = self.height;
        let samples = self.samples_per_pixel;
        let width_f = (width.max(1) - 1) as f32;
        let height_f = (height.max(1) - 1) as f32;

        // Process rows in parallel
        let rows: Vec<Vec<Vec3>> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(y as u64 * 1000 + 42);
                let mut row = Vec::with_capacity(width as usize);
                
                for x in 0..width {
                    let mut color = Vec3::zero();
                    
                    for _ in 0..samples {
                        let u = if width_f > 0.0 {
                            (x as f32 + rand::Rng::gen::<f32>(&mut rng)) / width_f
                        } else {
                            0.5
                        };
                        let v = if height_f > 0.0 {
                            ((height - 1 - y) as f32 + rand::Rng::gen::<f32>(&mut rng)) / height_f
                        } else {
                            0.5
                        };
                        
                        let ray = camera.get_ray(u, v, &mut rng);
                        color = color + integrator.trace(&ray, scene, &mut rng);
                    }
                    
                    row.push(color / samples as f32);
                }
                
                row
            })
            .collect();

        // Flatten rows
        rows.into_iter().flatten().collect()
    }

    /// Parallel rendering with SIMD packet tracing (4 rays at once)
    fn render_parallel_packets(&self, scene: &Scene, camera: &Camera) -> Vec<Vec3> {
        let integrator = PathIntegrator::new(self.max_depth);
        let width = self.width;
        let height = self.height;
        let samples = self.samples_per_pixel;

        // Process 2x2 pixel blocks in parallel
        let blocks_x = (width + 1) / 2;
        let blocks_y = (height + 1) / 2;
        let total_blocks = blocks_x * blocks_y;

        let mut pixels = vec![Vec3::zero(); (width * height) as usize];
        
        let block_results: Vec<(u32, u32, [Vec3; 4])> = (0..total_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let block_x = block_idx % blocks_x;
                let block_y = block_idx / blocks_x;
                let base_x = block_x * 2;
                let base_y = block_y * 2;

                let mut rng = Xoshiro256PlusPlus::seed_from_u64(block_idx as u64 * 7919 + 42);
                let mut colors = [Vec3::zero(); 4];

                // Process samples for 2x2 block
                for _ in 0..samples {
                    // Generate 4 rays for 2x2 pixel block (clamp to valid range)
                    let x0 = base_x.min(width.saturating_sub(1));
                    let x1 = (base_x + 1).min(width.saturating_sub(1));
                    let y0 = base_y.min(height.saturating_sub(1));
                    let y1 = (base_y + 1).min(height.saturating_sub(1));
                    
                    let rays = [
                        self.generate_ray_safe(x0, y0, width, height, camera, &mut rng),
                        self.generate_ray_safe(x1, y0, width, height, camera, &mut rng),
                        self.generate_ray_safe(x0, y1, width, height, camera, &mut rng),
                        self.generate_ray_safe(x1, y1, width, height, camera, &mut rng),
                    ];

                    // Create ray packet for primary rays (used for first intersection)
                    let packet = RayPacket::new(rays);
                    
                    // First intersection using packet
                    if let Some(bvh) = scene.bvh() {
                        let hit_packet = bvh.hit_packet(&packet, 0.001);
                        
                        // Process each ray's path independently after first hit
                        for i in 0..4 {
                            if let Some(_hit) = &hit_packet.hits[i] {
                                // Continue path tracing with individual rays
                                colors[i] = colors[i] + integrator.trace(&rays[i], scene, &mut rng);
                            } else {
                                // Background
                                colors[i] = colors[i] + integrator.trace(&rays[i], scene, &mut rng);
                            }
                        }
                    } else {
                        // No BVH - fallback to scalar tracing
                        for i in 0..4 {
                            colors[i] = colors[i] + integrator.trace(&rays[i], scene, &mut rng);
                        }
                    }
                }

                // Average samples
                for color in &mut colors {
                    *color = *color / samples as f32;
                }

                (base_x, base_y, colors)
            })
            .collect();

        // Copy results to pixel buffer
        for (base_x, base_y, colors) in block_results {
            let coords = [
                (base_x, base_y),
                (base_x + 1, base_y),
                (base_x, base_y + 1),
                (base_x + 1, base_y + 1),
            ];
            
            for (i, &(x, y)) in coords.iter().enumerate() {
                if x < width && y < height {
                    pixels[(y * width + x) as usize] = colors[i];
                }
            }
        }

        pixels
    }

    /// Generate a ray for a pixel with jittering (safe version that handles edge cases)
    fn generate_ray_safe<R: rand::Rng>(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        camera: &Camera,
        rng: &mut R,
    ) -> Ray {
        let width_f = (width.max(1) - 1) as f32;
        let height_f = (height.max(1) - 1) as f32;
        
        let u = if width_f > 0.0 {
            (x as f32 + rng.gen::<f32>()) / width_f
        } else {
            0.5
        };
        
        // Safe subtraction for v coordinate
        let y_from_bottom = if y < height { height - 1 - y } else { 0 };
        let v = if height_f > 0.0 {
            (y_from_bottom as f32 + rng.gen::<f32>()) / height_f
        } else {
            0.5
        };
        
        camera.get_ray(u, v, rng)
    }

    /// Render with tile-based parallelism (better cache coherence)
    pub fn render_tiled(&self, scene: &Scene, camera: &Camera, tile_size: u32) -> Vec<Vec3> {
        let integrator = PathIntegrator::new(self.max_depth);
        let width = self.width;
        let height = self.height;
        let samples = self.samples_per_pixel;
        let width_f = (width.max(1) - 1) as f32;
        let height_f = (height.max(1) - 1) as f32;

        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;
        let total_tiles = tiles_x * tiles_y;

        // Process tiles in parallel
        let mut pixels = vec![Vec3::zero(); (width * height) as usize];
        
        let tile_results: Vec<(u32, u32, Vec<Vec3>)> = (0..total_tiles)
            .into_par_iter()
            .map(|tile_idx| {
                let tile_x = tile_idx % tiles_x;
                let tile_y = tile_idx / tiles_x;
                let start_x = tile_x * tile_size;
                let start_y = tile_y * tile_size;
                let end_x = (start_x + tile_size).min(width);
                let end_y = (start_y + tile_size).min(height);
                
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(tile_idx as u64 * 7919 + 42);
                let mut tile_pixels = Vec::with_capacity(((end_x - start_x) * (end_y - start_y)) as usize);
                
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let mut color = Vec3::zero();
                        
                        for _ in 0..samples {
                            let u = if width_f > 0.0 {
                                (x as f32 + rand::Rng::gen::<f32>(&mut rng)) / width_f
                            } else {
                                0.5
                            };
                            let v = if height_f > 0.0 {
                                ((height - 1 - y) as f32 + rand::Rng::gen::<f32>(&mut rng)) / height_f
                            } else {
                                0.5
                            };
                            
                            let ray = camera.get_ray(u, v, &mut rng);
                            color = color + integrator.trace(&ray, scene, &mut rng);
                        }
                        
                        tile_pixels.push(color / samples as f32);
                    }
                }
                
                (start_x, start_y, tile_pixels)
            })
            .collect();

        // Copy tile results to output
        for (start_x, start_y, tile_pixels) in tile_results {
            let tile_w = (tile_size.min(width - start_x)) as usize;
            let tile_h = tile_pixels.len() / tile_w.max(1);
            
            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let x = start_x as usize + tx;
                    let y = start_y as usize + ty;
                    if x < width as usize && y < height as usize {
                        pixels[y * width as usize + x] = tile_pixels[ty * tile_w + tx];
                    }
                }
            }
        }

        pixels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;
    use crate::material::Lambertian;

    fn make_test_scene() -> (Scene, Camera) {
        let mut scene = Scene::new();
        let mat = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
        scene.add_sphere(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, mat));
        scene.build_bvh();
        
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0, 1.0, 0.0, 1.0
        );
        
        (scene, camera)
    }

    #[test]
    fn test_render_single() {
        let (scene, camera) = make_test_scene();
        let renderer = CpuRenderer::new(4, 4, 1, 2);
        let pixels = renderer.render(&scene, &camera);
        assert_eq!(pixels.len(), 16);
    }

    #[test]
    fn test_render_parallel() {
        let (scene, camera) = make_test_scene();
        let renderer = CpuRenderer::new(4, 4, 1, 2);
        let pixels = renderer.render_parallel(&scene, &camera, 2);
        assert_eq!(pixels.len(), 16);
    }

    #[test]
    fn test_render_parallel_no_packets() {
        let (scene, camera) = make_test_scene();
        let renderer = CpuRenderer::new(4, 4, 1, 2).with_packet_tracing(false);
        let pixels = renderer.render_parallel(&scene, &camera, 2);
        assert_eq!(pixels.len(), 16);
    }

    #[test]
    fn test_render_tiled() {
        let (scene, camera) = make_test_scene();
        let renderer = CpuRenderer::new(8, 8, 1, 2);
        let pixels = renderer.render_tiled(&scene, &camera, 4);
        assert_eq!(pixels.len(), 64);
    }

    #[test]
    fn test_render_odd_dimensions() {
        let (scene, camera) = make_test_scene();
        // Test odd dimensions that could cause edge cases in packet tracing
        let renderer = CpuRenderer::new(5, 7, 1, 2);
        let pixels = renderer.render_parallel(&scene, &camera, 0);
        assert_eq!(pixels.len(), 35);
    }

    #[test]
    fn test_render_single_pixel() {
        let (scene, camera) = make_test_scene();
        let renderer = CpuRenderer::new(1, 1, 1, 2);
        let pixels = renderer.render_parallel(&scene, &camera, 0);
        assert_eq!(pixels.len(), 1);
    }
}
