//! Heterogeneous participating medium - spatially varying density.
//!
//! Uses a 3D grid of density values and delta tracking (Woodcock tracking)
//! for unbiased sampling through the volume.

use super::{PhaseFunction, Transmittance, Volume, VolumeSample, HenyeyGreenstein};
use crate::math::{Vec3, Ray};
use rand::Rng;

/// Heterogeneous volume with grid-based density.
#[derive(Debug, Clone)]
pub struct HeterogeneousVolume {
    /// 3D density grid (values in [0, 1] relative to max_density).
    grid: Vec<f32>,
    /// Grid dimensions.
    nx: usize,
    ny: usize,
    nz: usize,
    /// World-space bounds of the volume.
    min: Vec3,
    max: Vec3,
    /// Maximum density value (for delta tracking).
    max_density: f32,
    /// Base extinction coefficient (scaled by density).
    sigma_t_base: Vec3,
    /// Single-scattering albedo.
    albedo: Vec3,
    /// Phase function.
    phase: HenyeyGreenstein,
}

impl HeterogeneousVolume {
    /// Create a new heterogeneous volume from a density grid.
    ///
    /// # Arguments
    /// * `grid` - Flat array of density values (nx * ny * nz), normalized to [0, 1]
    /// * `dims` - Grid dimensions (nx, ny, nz)
    /// * `bounds` - World-space (min, max) bounds
    /// * `max_density` - Peak extinction coefficient
    /// * `albedo` - Single-scattering albedo
    /// * `g` - Asymmetry parameter for phase function
    pub fn new(
        grid: Vec<f32>,
        dims: (usize, usize, usize),
        bounds: (Vec3, Vec3),
        max_density: f32,
        albedo: Vec3,
        g: f32,
    ) -> Self {
        assert_eq!(grid.len(), dims.0 * dims.1 * dims.2);
        
        Self {
            grid,
            nx: dims.0,
            ny: dims.1,
            nz: dims.2,
            min: bounds.0,
            max: bounds.1,
            max_density,
            sigma_t_base: Vec3::new(max_density, max_density, max_density),
            albedo,
            phase: HenyeyGreenstein::new(g),
        }
    }

    /// Create a sphere of varying density (dense center, thin edges).
    pub fn sphere(center: Vec3, radius: f32, resolution: usize, max_density: f32) -> Self {
        let half = Vec3::new(radius, radius, radius);
        let min = center - half;
        let max = center + half;
        
        let n = resolution;
        let mut grid = vec![0.0; n * n * n];
        
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    // Normalized position in grid
                    let px = (x as f32 + 0.5) / n as f32;
                    let py = (y as f32 + 0.5) / n as f32;
                    let pz = (z as f32 + 0.5) / n as f32;
                    
                    // Distance from center (0 to 1)
                    let dx = px - 0.5;
                    let dy = py - 0.5;
                    let dz = pz - 0.5;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt() * 2.0;
                    
                    // Falloff: dense at center, zero at edge
                    let density = (1.0 - dist).max(0.0).powf(2.0);
                    grid[z * n * n + y * n + x] = density;
                }
            }
        }
        
        Self::new(
            grid,
            (n, n, n),
            (min, max),
            max_density,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        )
    }

    /// Create a procedural noise-based cloud volume.
    pub fn noise_cloud(center: Vec3, size: Vec3, resolution: usize, max_density: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        
        let min = center - size * 0.5;
        let max = center + size * 0.5;
        
        let n = resolution;
        let mut grid = vec![0.0; n * n * n];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        
        // Simple 3D value noise
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    // Base noise
                    let noise: f32 = rng.gen();
                    
                    // Fade at edges
                    let fx = (x as f32 / n as f32 - 0.5).abs() * 2.0;
                    let fy = (y as f32 / n as f32 - 0.5).abs() * 2.0;
                    let fz = (z as f32 / n as f32 - 0.5).abs() * 2.0;
                    let edge_fade = (1.0 - fx.max(fy).max(fz)).max(0.0);
                    
                    grid[z * n * n + y * n + x] = noise * edge_fade * edge_fade;
                }
            }
        }
        
        Self::new(
            grid,
            (n, n, n),
            (min, max),
            max_density,
            Vec3::new(0.95, 0.95, 0.95),
            0.85, // Forward scattering like clouds
        )
    }

    /// Sample density at a world-space point using trilinear interpolation.
    fn density_at(&self, point: &Vec3) -> f32 {
        // Convert world to grid coordinates
        let range = self.max - self.min;
        let offset = *point - self.min;
        let local = Vec3::new(
            offset.x / range.x,
            offset.y / range.y,
            offset.z / range.z,
        );
        
        // Check bounds
        if local.x < 0.0 || local.x >= 1.0 ||
           local.y < 0.0 || local.y >= 1.0 ||
           local.z < 0.0 || local.z >= 1.0 {
            return 0.0;
        }
        
        // Grid coordinates
        let gx = local.x * (self.nx - 1) as f32;
        let gy = local.y * (self.ny - 1) as f32;
        let gz = local.z * (self.nz - 1) as f32;
        
        let x0 = (gx as usize).min(self.nx - 2);
        let y0 = (gy as usize).min(self.ny - 2);
        let z0 = (gz as usize).min(self.nz - 2);
        
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let fz = gz - z0 as f32;
        
        // Trilinear interpolation
        let idx = |x, y, z| z * self.nx * self.ny + y * self.nx + x;
        
        let c000 = self.grid[idx(x0, y0, z0)];
        let c100 = self.grid[idx(x0 + 1, y0, z0)];
        let c010 = self.grid[idx(x0, y0 + 1, z0)];
        let c110 = self.grid[idx(x0 + 1, y0 + 1, z0)];
        let c001 = self.grid[idx(x0, y0, z0 + 1)];
        let c101 = self.grid[idx(x0 + 1, y0, z0 + 1)];
        let c011 = self.grid[idx(x0, y0 + 1, z0 + 1)];
        let c111 = self.grid[idx(x0 + 1, y0 + 1, z0 + 1)];
        
        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;
        
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        
        c0 * (1.0 - fz) + c1 * fz
    }

    /// Check if a point is inside the volume bounds.
    fn contains(&self, point: &Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }

    /// Intersect ray with volume bounding box, returns (t_enter, t_exit).
    fn intersect_bounds(&self, ray: &Ray) -> Option<(f32, f32)> {
        let inv_dir = Vec3::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        
        let t1 = (self.min.x - ray.origin.x) * inv_dir.x;
        let t2 = (self.max.x - ray.origin.x) * inv_dir.x;
        let t3 = (self.min.y - ray.origin.y) * inv_dir.y;
        let t4 = (self.max.y - ray.origin.y) * inv_dir.y;
        let t5 = (self.min.z - ray.origin.z) * inv_dir.z;
        let t6 = (self.max.z - ray.origin.z) * inv_dir.z;
        
        let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));
        
        if t_max < 0.0 || t_min > t_max {
            None
        } else {
            Some((t_min.max(0.0), t_max))
        }
    }
}

impl Volume for HeterogeneousVolume {
    fn sigma_t(&self, point: &Vec3) -> Vec3 {
        let d = self.density_at(point);
        self.sigma_t_base * d
    }

    fn sigma_s(&self, point: &Vec3) -> Vec3 {
        let st = self.sigma_t(point);
        Vec3::new(st.x * self.albedo.x, st.y * self.albedo.y, st.z * self.albedo.z)
    }

    fn max_sigma_t(&self) -> f32 {
        self.max_density
    }

    fn sample(&self, ray: &Ray, t_max: f32, rng: &mut impl Rng) -> VolumeSample {
        // Intersect with volume bounds
        let (t_enter, t_exit) = match self.intersect_bounds(ray) {
            Some((enter, exit)) => (enter.max(0.0), exit.min(t_max)),
            None => {
                return VolumeSample {
                    t: t_max,
                    scattered: false,
                    transmittance: Vec3::one(),
                    position: ray.at(t_max),
                };
            }
        };
        
        if t_enter >= t_exit {
            return VolumeSample {
                t: t_max,
                scattered: false,
                transmittance: Vec3::one(),
                position: ray.at(t_max),
            };
        }
        
        // Delta tracking (Woodcock tracking)
        let mut t = t_enter;
        let inv_max_sigma = 1.0 / self.max_density;
        
        loop {
            // Sample free-flight distance with majorant
            let u: f32 = rng.gen();
            t += -u.ln() * inv_max_sigma;
            
            if t >= t_exit {
                // Exited the volume
                let transmittance = self.transmittance(ray, t_exit - t_enter, rng);
                return VolumeSample {
                    t: t_exit,
                    scattered: false,
                    transmittance: transmittance.value,
                    position: ray.at(t_exit),
                };
            }
            
            // Evaluate density at sample point
            let point = ray.at(t);
            let density = self.density_at(&point);
            
            // Accept/reject based on density ratio
            if rng.gen::<f32>() < density {
                // Real scattering event
                let transmittance = self.transmittance(ray, t - t_enter, rng);
                return VolumeSample {
                    t,
                    scattered: true,
                    transmittance: transmittance.value,
                    position: point,
                };
            }
            // Null collision - continue tracking
        }
    }

    fn transmittance(&self, ray: &Ray, t_max: f32, rng: &mut impl Rng) -> Transmittance {
        // Ratio tracking for transmittance estimation
        let (t_enter, t_exit) = match self.intersect_bounds(ray) {
            Some((enter, exit)) => (enter.max(0.0), exit.min(t_max)),
            None => return Transmittance::one(),
        };
        
        if t_enter >= t_exit {
            return Transmittance::one();
        }
        
        let mut tr = 1.0_f32;
        let mut t = t_enter;
        let inv_max_sigma = 1.0 / self.max_density;
        
        loop {
            let u: f32 = rng.gen();
            t += -u.ln() * inv_max_sigma;
            
            if t >= t_exit {
                break;
            }
            
            let point = ray.at(t);
            let density = self.density_at(&point);
            tr *= 1.0 - density;
            
            // Russian roulette termination
            if tr < 0.1 {
                if rng.gen::<f32>() > tr {
                    return Transmittance { value: Vec3::zero() };
                }
                tr = 1.0;
            }
        }
        
        Transmittance {
            value: Vec3::new(tr, tr, tr),
        }
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

    fn make_uniform_grid(value: f32, n: usize) -> Vec<f32> {
        vec![value; n * n * n]
    }

    #[test]
    fn test_heterogeneous_creation() {
        let grid = make_uniform_grid(0.5, 4);
        let vol = HeterogeneousVolume::new(
            grid,
            (4, 4, 4),
            (Vec3::zero(), Vec3::one()),
            1.0,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        assert_eq!(vol.nx, 4);
        assert!((vol.max_sigma_t() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_density_interpolation() {
        let mut grid = make_uniform_grid(0.0, 4);
        // Set one corner to 1.0
        grid[0] = 1.0;
        
        let vol = HeterogeneousVolume::new(
            grid,
            (4, 4, 4),
            (Vec3::zero(), Vec3::one()),
            1.0,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        // Near the corner should have some density
        let near_corner = vol.density_at(&Vec3::new(0.1, 0.1, 0.1));
        let far_corner = vol.density_at(&Vec3::new(0.9, 0.9, 0.9));
        
        assert!(near_corner > far_corner);
    }

    #[test]
    fn test_outside_bounds_zero_density() {
        let grid = make_uniform_grid(1.0, 4);
        let vol = HeterogeneousVolume::new(
            grid,
            (4, 4, 4),
            (Vec3::zero(), Vec3::one()),
            1.0,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        assert!((vol.density_at(&Vec3::new(-1.0, 0.5, 0.5))).abs() < 1e-6);
        assert!((vol.density_at(&Vec3::new(2.0, 0.5, 0.5))).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_volume() {
        let vol = HeterogeneousVolume::sphere(Vec3::zero(), 2.0, 16, 0.5);
        
        // Center should have highest density
        let center = vol.density_at(&Vec3::zero());
        let edge = vol.density_at(&Vec3::new(0.9, 0.0, 0.0));
        
        assert!(center > edge, "center {} should be denser than edge {}", center, edge);
    }

    #[test]
    fn test_noise_cloud() {
        let vol = HeterogeneousVolume::noise_cloud(
            Vec3::zero(),
            Vec3::new(2.0, 2.0, 2.0),
            8,
            0.3,
            42,
        );
        
        // Should have varying density
        let mut densities = Vec::new();
        for _ in 0..10 {
            let p = Vec3::new(
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
            );
            densities.push(vol.density_at(&p));
        }
        
        // Not all same
        let first = densities[0];
        assert!(densities.iter().any(|&d| (d - first).abs() > 0.01));
    }

    #[test]
    fn test_sample_through_volume() {
        let vol = HeterogeneousVolume::sphere(Vec3::zero(), 2.0, 8, 0.5);
        let ray = Ray::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let mut scattered = 0;
        let mut exited = 0;
        
        for _ in 0..100 {
            let sample = vol.sample(&ray, 10.0, &mut rng);
            if sample.scattered {
                scattered += 1;
            } else {
                exited += 1;
            }
        }
        
        // Should have both outcomes
        assert!(scattered > 0 || exited > 0);
    }

    #[test]
    fn test_transmittance_decreases() {
        let grid = make_uniform_grid(1.0, 4);
        let vol = HeterogeneousVolume::new(
            grid,
            (4, 4, 4),
            (Vec3::zero(), Vec3::one()),
            0.5,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        let ray = Ray::new(Vec3::new(0.1, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        // Average over multiple samples (stochastic)
        let mut sum_short = 0.0;
        let mut sum_long = 0.0;
        for _ in 0..100 {
            sum_short += vol.transmittance(&ray, 0.3, &mut rng).value.x;
            sum_long += vol.transmittance(&ray, 0.8, &mut rng).value.x;
        }
        
        assert!(sum_short > sum_long, "short path should transmit more");
    }

    #[test]
    fn test_bounds_intersection() {
        let grid = make_uniform_grid(0.5, 4);
        let vol = HeterogeneousVolume::new(
            grid,
            (4, 4, 4),
            (Vec3::zero(), Vec3::one()),
            1.0,
            Vec3::new(0.9, 0.9, 0.9),
            0.0,
        );
        
        // Ray through volume
        let ray_through = Ray::new(Vec3::new(-1.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(vol.intersect_bounds(&ray_through).is_some());
        
        // Ray missing volume
        let ray_miss = Ray::new(Vec3::new(-1.0, 5.0, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(vol.intersect_bounds(&ray_miss).is_none());
    }
}
