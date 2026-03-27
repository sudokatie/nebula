//! Perlin noise texture

use crate::math::Vec3;
use super::Texture;
use rand::prelude::*;

const POINT_COUNT: usize = 256;

/// Perlin noise generator
pub struct Perlin {
    ranvec: [Vec3; POINT_COUNT],
    perm_x: [usize; POINT_COUNT],
    perm_y: [usize; POINT_COUNT],
    perm_z: [usize; POINT_COUNT],
}

impl Perlin {
    /// Create new Perlin noise generator
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let mut ranvec = [Vec3::zero(); POINT_COUNT];
        for v in ranvec.iter_mut() {
            *v = Vec3::random_range(&mut rng, -1.0, 1.0).normalize();
        }

        Self {
            ranvec,
            perm_x: Self::generate_perm(&mut rng),
            perm_y: Self::generate_perm(&mut rng),
            perm_z: Self::generate_perm(&mut rng),
        }
    }

    fn generate_perm<R: Rng>(rng: &mut R) -> [usize; POINT_COUNT] {
        let mut perm: [usize; POINT_COUNT] = std::array::from_fn(|i| i);

        // Fisher-Yates shuffle
        for i in (1..POINT_COUNT).rev() {
            let target = rng.gen_range(0..=i);
            perm.swap(i, target);
        }

        perm
    }

    /// Sample noise at point
    pub fn noise(&self, point: &Vec3) -> f32 {
        let u = point.x - point.x.floor();
        let v = point.y - point.y.floor();
        let w = point.z - point.z.floor();

        let i = point.x.floor() as i32;
        let j = point.y.floor() as i32;
        let k = point.z.floor() as i32;

        let mut c = [[[Vec3::zero(); 2]; 2]; 2];

        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    let idx = self.perm_x[((i + di as i32) & 255) as usize]
                        ^ self.perm_y[((j + dj as i32) & 255) as usize]
                        ^ self.perm_z[((k + dk as i32) & 255) as usize];
                    c[di][dj][dk] = self.ranvec[idx];
                }
            }
        }

        Self::trilinear_interp(&c, u, v, w)
    }

    /// Turbulence (sum of multiple octaves)
    pub fn turbulence(&self, point: &Vec3, depth: u32) -> f32 {
        let mut accum = 0.0;
        let mut temp_p = *point;
        let mut weight = 1.0;

        for _ in 0..depth {
            accum += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p = temp_p * 2.0;
        }

        accum.abs()
    }

    fn trilinear_interp(c: &[[[Vec3; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
        // Hermite cubic for smoothing
        let uu = u * u * (3.0 - 2.0 * u);
        let vv = v * v * (3.0 - 2.0 * v);
        let ww = w * w * (3.0 - 2.0 * w);

        let mut accum = 0.0;

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let fi = i as f32;
                    let fj = j as f32;
                    let fk = k as f32;

                    let weight = Vec3::new(u - fi, v - fj, w - fk);

                    accum += (fi * uu + (1.0 - fi) * (1.0 - uu))
                        * (fj * vv + (1.0 - fj) * (1.0 - vv))
                        * (fk * ww + (1.0 - fk) * (1.0 - ww))
                        * c[i][j][k].dot(&weight);
                }
            }
        }

        accum
    }
}

impl Default for Perlin {
    fn default() -> Self {
        Self::new()
    }
}

/// Noise texture using Perlin noise
pub struct NoiseTexture {
    perlin: Perlin,
    scale: f32,
    turbulence_depth: u32,
    use_marble: bool,
}

impl NoiseTexture {
    /// Create basic noise texture
    pub fn new(scale: f32) -> Self {
        Self {
            perlin: Perlin::new(),
            scale,
            turbulence_depth: 7,
            use_marble: false,
        }
    }

    /// Create turbulent noise
    pub fn turbulent(scale: f32, depth: u32) -> Self {
        Self {
            perlin: Perlin::new(),
            scale,
            turbulence_depth: depth,
            use_marble: false,
        }
    }

    /// Create marble-like texture
    pub fn marble(scale: f32) -> Self {
        Self {
            perlin: Perlin::new(),
            scale,
            turbulence_depth: 7,
            use_marble: true,
        }
    }
}

impl Texture for NoiseTexture {
    fn value(&self, _u: f32, _v: f32, point: &Vec3) -> Vec3 {
        let scaled = *point * self.scale;

        let noise_val = if self.use_marble {
            // Marble pattern using sine with turbulence
            0.5 * (1.0 + (scaled.z + 10.0 * self.perlin.turbulence(&scaled, self.turbulence_depth)).sin())
        } else {
            // Basic turbulence
            self.perlin.turbulence(&scaled, self.turbulence_depth)
        };

        Vec3::one() * noise_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_noise_range() {
        let perlin = Perlin::new();
        for x in 0..10 {
            for y in 0..10 {
                let p = Vec3::new(x as f32 * 0.1, y as f32 * 0.1, 0.0);
                let n = perlin.noise(&p);
                // Perlin noise should be roughly in [-1, 1]
                assert!(n >= -1.5 && n <= 1.5, "noise {} out of expected range", n);
            }
        }
    }

    #[test]
    fn test_perlin_continuous() {
        let perlin = Perlin::new();
        let p1 = Vec3::new(0.5, 0.5, 0.5);
        let p2 = Vec3::new(0.501, 0.5, 0.5);
        let n1 = perlin.noise(&p1);
        let n2 = perlin.noise(&p2);
        // Should be similar for nearby points
        assert!((n1 - n2).abs() < 0.1);
    }

    #[test]
    fn test_turbulence() {
        let perlin = Perlin::new();
        let p = Vec3::new(1.0, 2.0, 3.0);
        let t = perlin.turbulence(&p, 7);
        // Turbulence should be non-negative
        assert!(t >= 0.0);
    }

    #[test]
    fn test_noise_texture() {
        let tex = NoiseTexture::new(1.0);
        let color = tex.value(0.0, 0.0, &Vec3::new(1.0, 2.0, 3.0));
        // Should produce grayscale value
        assert!((color.x - color.y).abs() < 1e-6);
        assert!((color.y - color.z).abs() < 1e-6);
    }

    #[test]
    fn test_marble_texture() {
        let tex = NoiseTexture::marble(1.0);
        let color = tex.value(0.0, 0.0, &Vec3::new(1.0, 2.0, 3.0));
        // Marble should be in [0, 1] due to sine normalization
        assert!(color.x >= 0.0 && color.x <= 1.0);
    }
}
