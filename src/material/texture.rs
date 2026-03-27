//! Texture support with LOD/mipmap filtering

use crate::math::Vec3;
use std::sync::Arc;

/// Texture trait for spatially-varying properties
pub trait Texture: Send + Sync {
    /// Sample texture at UV coordinates (default LOD)
    fn sample(&self, u: f32, v: f32, point: &Vec3) -> Vec3;
    
    /// Sample texture with LOD (level of detail) for mipmap filtering
    /// footprint: approximate size of the texture filter region in texture space
    fn sample_lod(&self, u: f32, v: f32, point: &Vec3, footprint: f32) -> Vec3 {
        // Default: ignore LOD and use base sample
        let _ = footprint;
        self.sample(u, v, point)
    }
}

/// Solid color texture
#[derive(Clone)]
pub struct SolidColor {
    color: Vec3,
}

impl SolidColor {
    pub fn new(color: Vec3) -> Self {
        Self { color }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(Vec3::new(r, g, b))
    }
}

impl Texture for SolidColor {
    fn sample(&self, _u: f32, _v: f32, _point: &Vec3) -> Vec3 {
        self.color
    }
}

/// Checkerboard texture
#[derive(Clone)]
pub struct Checker {
    odd: Arc<dyn Texture>,
    even: Arc<dyn Texture>,
    scale: f32,
}

impl Checker {
    pub fn new(odd: Arc<dyn Texture>, even: Arc<dyn Texture>, scale: f32) -> Self {
        Self { odd, even, scale }
    }

    pub fn colors(c1: Vec3, c2: Vec3, scale: f32) -> Self {
        Self::new(
            Arc::new(SolidColor::new(c1)),
            Arc::new(SolidColor::new(c2)),
            scale,
        )
    }
}

impl Texture for Checker {
    fn sample(&self, u: f32, v: f32, point: &Vec3) -> Vec3 {
        let sines = (self.scale * point.x).sin()
            * (self.scale * point.y).sin()
            * (self.scale * point.z).sin();
        
        if sines < 0.0 {
            self.odd.sample(u, v, point)
        } else {
            self.even.sample(u, v, point)
        }
    }
}

/// Image texture with mipmap support
pub struct ImageTexture {
    /// Mipmap levels (level 0 = full resolution)
    mip_levels: Vec<MipLevel>,
    width: u32,
    height: u32,
}

struct MipLevel {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

impl ImageTexture {
    pub fn new(data: Vec<u8>, width: u32, height: u32, bytes_per_pixel: u32) -> Self {
        // Convert to RGB if needed and build mipmaps
        let rgb_data = if bytes_per_pixel == 4 {
            // RGBA -> RGB
            data.chunks(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect()
        } else {
            data
        };
        
        let mut mip_levels = vec![MipLevel {
            data: rgb_data,
            width,
            height,
        }];
        
        // Generate mipmap chain
        let mut mip_width = width;
        let mut mip_height = height;
        
        while mip_width > 1 || mip_height > 1 {
            let prev = mip_levels.last().unwrap();
            let new_width = (mip_width / 2).max(1);
            let new_height = (mip_height / 2).max(1);
            
            let mut new_data = Vec::with_capacity((new_width * new_height * 3) as usize);
            
            for j in 0..new_height {
                for i in 0..new_width {
                    // Box filter (average 2x2 pixels)
                    let x0 = (i * 2).min(prev.width - 1);
                    let x1 = ((i * 2) + 1).min(prev.width - 1);
                    let y0 = (j * 2).min(prev.height - 1);
                    let y1 = ((j * 2) + 1).min(prev.height - 1);
                    
                    for c in 0..3 {
                        let p00 = prev.data[((y0 * prev.width + x0) * 3 + c) as usize] as u32;
                        let p01 = prev.data[((y0 * prev.width + x1) * 3 + c) as usize] as u32;
                        let p10 = prev.data[((y1 * prev.width + x0) * 3 + c) as usize] as u32;
                        let p11 = prev.data[((y1 * prev.width + x1) * 3 + c) as usize] as u32;
                        let avg = ((p00 + p01 + p10 + p11) / 4) as u8;
                        new_data.push(avg);
                    }
                }
            }
            
            mip_levels.push(MipLevel {
                data: new_data,
                width: new_width,
                height: new_height,
            });
            
            mip_width = new_width;
            mip_height = new_height;
        }
        
        Self {
            mip_levels,
            width,
            height,
        }
    }

    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let img = image::open(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        Ok(Self::new(rgb.into_raw(), width, height, 3))
    }

    /// Sample with bilinear filtering at a specific mip level
    fn sample_bilinear(&self, u: f32, v: f32, level: usize) -> Vec3 {
        let level = level.min(self.mip_levels.len() - 1);
        let mip = &self.mip_levels[level];
        
        let u = u.clamp(0.0, 1.0);
        let v = 1.0 - v.clamp(0.0, 1.0); // Flip V
        
        let x = u * (mip.width - 1) as f32;
        let y = v * (mip.height - 1) as f32;
        
        let x0 = (x.floor() as u32).min(mip.width - 1);
        let y0 = (y.floor() as u32).min(mip.height - 1);
        let x1 = (x0 + 1).min(mip.width - 1);
        let y1 = (y0 + 1).min(mip.height - 1);
        
        let fx = x - x.floor();
        let fy = y - y.floor();
        
        let scale = 1.0 / 255.0;
        
        let sample = |i: u32, j: u32| -> Vec3 {
            let idx = ((j * mip.width + i) * 3) as usize;
            Vec3::new(
                mip.data[idx] as f32 * scale,
                mip.data[idx + 1] as f32 * scale,
                mip.data[idx + 2] as f32 * scale,
            )
        };
        
        let c00 = sample(x0, y0);
        let c10 = sample(x1, y0);
        let c01 = sample(x0, y1);
        let c11 = sample(x1, y1);
        
        // Bilinear interpolation
        let c0 = c00 * (1.0 - fx) + c10 * fx;
        let c1 = c01 * (1.0 - fx) + c11 * fx;
        c0 * (1.0 - fy) + c1 * fy
    }
}

impl Texture for ImageTexture {
    fn sample(&self, u: f32, v: f32, _point: &Vec3) -> Vec3 {
        self.sample_bilinear(u, v, 0)
    }
    
    fn sample_lod(&self, u: f32, v: f32, _point: &Vec3, footprint: f32) -> Vec3 {
        // Compute LOD from footprint
        // footprint is the approximate size of the filter region in world space
        // We need to convert this to mip level
        let texels_per_unit = self.width.max(self.height) as f32;
        let lod = (footprint * texels_per_unit).max(1.0).log2();
        let lod_floor = lod.floor() as usize;
        let lod_ceil = (lod_floor + 1).min(self.mip_levels.len() - 1);
        let lod_frac = lod - lod.floor();
        
        // Trilinear filtering (blend between mip levels)
        let c0 = self.sample_bilinear(u, v, lod_floor);
        let c1 = self.sample_bilinear(u, v, lod_ceil);
        c0 * (1.0 - lod_frac) + c1 * lod_frac
    }
}

/// Noise texture using simple value noise
pub struct NoiseTexture {
    scale: f32,
}

impl NoiseTexture {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    fn noise(&self, p: Vec3) -> f32 {
        // Simple hash-based noise
        let hash = |x: i32, y: i32, z: i32| -> f32 {
            let n = x.wrapping_add(y.wrapping_mul(57)).wrapping_add(z.wrapping_mul(131));
            let n = (n << 13) ^ n;
            let m = n.wrapping_mul(n.wrapping_mul(n.wrapping_mul(15731).wrapping_add(789221)).wrapping_add(1376312589));
            1.0 - (m & 0x7fffffff) as f32 / 1073741824.0
        };

        let px = (p.x * self.scale).floor() as i32;
        let py = (p.y * self.scale).floor() as i32;
        let pz = (p.z * self.scale).floor() as i32;

        let fx = p.x * self.scale - px as f32;
        let fy = p.y * self.scale - py as f32;
        let fz = p.z * self.scale - pz as f32;

        let ux = fx * fx * (3.0 - 2.0 * fx);
        let uy = fy * fy * (3.0 - 2.0 * fy);
        let uz = fz * fz * (3.0 - 2.0 * fz);

        let n000 = hash(px, py, pz);
        let n001 = hash(px, py, pz + 1);
        let n010 = hash(px, py + 1, pz);
        let n011 = hash(px, py + 1, pz + 1);
        let n100 = hash(px + 1, py, pz);
        let n101 = hash(px + 1, py, pz + 1);
        let n110 = hash(px + 1, py + 1, pz);
        let n111 = hash(px + 1, py + 1, pz + 1);

        let nx00 = n000 * (1.0 - ux) + n100 * ux;
        let nx01 = n001 * (1.0 - ux) + n101 * ux;
        let nx10 = n010 * (1.0 - ux) + n110 * ux;
        let nx11 = n011 * (1.0 - ux) + n111 * ux;

        let nxy0 = nx00 * (1.0 - uy) + nx10 * uy;
        let nxy1 = nx01 * (1.0 - uy) + nx11 * uy;

        (nxy0 * (1.0 - uz) + nxy1 * uz) * 0.5 + 0.5
    }
}

impl Texture for NoiseTexture {
    fn sample(&self, _u: f32, _v: f32, point: &Vec3) -> Vec3 {
        let n = self.noise(*point);
        Vec3::new(n, n, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_color() {
        let tex = SolidColor::rgb(1.0, 0.0, 0.0);
        let c = tex.sample(0.0, 0.0, &Vec3::zero());
        assert_eq!(c.x, 1.0);
    }

    #[test]
    fn test_checker() {
        let tex = Checker::colors(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            10.0,
        );
        let c1 = tex.sample(0.0, 0.0, &Vec3::new(0.0, 0.0, 0.0));
        let c2 = tex.sample(0.0, 0.0, &Vec3::new(0.5, 0.5, 0.5));
        // Colors should be different at different points
        assert!((c1.x - c2.x).abs() > 0.1 || c1.x == c2.x);
    }

    #[test]
    fn test_noise() {
        let tex = NoiseTexture::new(1.0);
        let c = tex.sample(0.0, 0.0, &Vec3::new(0.5, 0.5, 0.5));
        assert!(c.x >= 0.0 && c.x <= 1.0);
    }
}
