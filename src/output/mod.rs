//! Image output with tone mapping

use crate::math::Vec3;
use std::io::{self, Write, BufWriter};
use std::fs::File;
use std::path::Path;

/// Tone mapping operators
#[derive(Debug, Clone, Copy, Default)]
pub enum ToneMap {
    /// No tone mapping (linear)
    #[default]
    Linear,
    /// Reinhard tone mapping
    Reinhard,
    /// ACES filmic
    Aces,
    /// Uncharted 2
    Uncharted2,
}

impl ToneMap {
    /// Apply tone mapping to HDR value
    pub fn apply(&self, color: Vec3) -> Vec3 {
        match self {
            ToneMap::Linear => color,
            ToneMap::Reinhard => {
                Vec3::new(
                    color.x / (color.x + 1.0),
                    color.y / (color.y + 1.0),
                    color.z / (color.z + 1.0),
                )
            }
            ToneMap::Aces => {
                // ACES filmic curve (per-channel)
                fn aces_channel(x: f32) -> f32 {
                    let a = 2.51;
                    let b = 0.03;
                    let c = 2.43;
                    let d = 0.59;
                    let e = 0.14;
                    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
                }
                Vec3::new(
                    aces_channel(color.x),
                    aces_channel(color.y),
                    aces_channel(color.z),
                )
            }
            ToneMap::Uncharted2 => {
                fn uncharted(x: f32) -> f32 {
                    let a = 0.15;
                    let b = 0.50;
                    let c = 0.10;
                    let d = 0.20;
                    let e = 0.02;
                    let f = 0.30;
                    ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
                }
                let white = 11.2;
                let white_scale = 1.0 / uncharted(white);
                Vec3::new(
                    uncharted(color.x * 2.0) * white_scale,
                    uncharted(color.y * 2.0) * white_scale,
                    uncharted(color.z * 2.0) * white_scale,
                )
            }
        }
    }
}

/// Apply exposure adjustment
pub fn apply_exposure(color: Vec3, exposure: f32) -> Vec3 {
    color * (2.0_f32).powf(exposure)
}

/// Convert linear color to sRGB
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB to linear
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Process pixel for output
pub fn process_pixel(color: Vec3, exposure: f32, tone_map: ToneMap) -> Vec3 {
    let exposed = apply_exposure(color, exposure);
    let mapped = tone_map.apply(exposed);
    Vec3::new(
        linear_to_srgb(mapped.x.clamp(0.0, 1.0)),
        linear_to_srgb(mapped.y.clamp(0.0, 1.0)),
        linear_to_srgb(mapped.z.clamp(0.0, 1.0)),
    )
}

/// Save image as PPM
pub fn save_ppm(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    save_ppm_with_options(path, pixels, width, height, 0.0, ToneMap::Linear)
}

/// Save image as PPM with tone mapping
pub fn save_ppm_with_options(
    path: &Path,
    pixels: &[Vec3],
    width: u32,
    height: u32,
    exposure: f32,
    tone_map: ToneMap,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "P3")?;
    writeln!(writer, "{} {}", width, height)?;
    writeln!(writer, "255")?;
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let color = process_pixel(pixels[idx], exposure, tone_map);
            let r = (color.x * 255.999) as u32;
            let g = (color.y * 255.999) as u32;
            let b = (color.z * 255.999) as u32;
            writeln!(writer, "{} {} {}", r, g, b)?;
        }
    }
    
    Ok(())
}

/// Save image as PNG
pub fn save_png(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    save_png_with_options(path, pixels, width, height, 0.0, ToneMap::Linear)
}

/// Save image as PNG with tone mapping
pub fn save_png_with_options(
    path: &Path,
    pixels: &[Vec3],
    width: u32,
    height: u32,
    exposure: f32,
    tone_map: ToneMap,
) -> io::Result<()> {
    let mut img = image::RgbImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let color = process_pixel(pixels[idx], exposure, tone_map);
            img.put_pixel(x, y, image::Rgb([
                (color.x * 255.999) as u8,
                (color.y * 255.999) as u8,
                (color.z * 255.999) as u8,
            ]));
        }
    }
    
    img.save(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Save HDR image in Radiance RGBE format
pub fn save_hdr(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writeln!(writer, "#?RADIANCE")?;
    writeln!(writer, "FORMAT=32-bit_rle_rgbe")?;
    writeln!(writer)?;
    writeln!(writer, "-Y {} +X {}", height, width)?;
    
    // Write pixels as RGBE
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let color = pixels[idx];
            let rgbe = float_to_rgbe(color);
            writer.write_all(&rgbe)?;
        }
    }
    
    Ok(())
}

/// Convert float RGB to RGBE
fn float_to_rgbe(color: Vec3) -> [u8; 4] {
    let max = color.x.max(color.y).max(color.z);
    
    if max < 1e-32 {
        return [0, 0, 0, 0];
    }
    
    let (mantissa, exp) = frexp(max);
    let scale = mantissa * 256.0 / max;
    
    [
        (color.x * scale) as u8,
        (color.y * scale) as u8,
        (color.z * scale) as u8,
        (exp + 128) as u8,
    ]
}

fn frexp(x: f32) -> (f32, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 126;
    let mantissa = f32::from_bits((bits & 0x807fffff) | 0x3f000000);
    (mantissa, exp)
}

/// Save EXR (OpenEXR format)
pub fn save_exr(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    use exr::prelude::*;
    
    let data: Vec<(f32, f32, f32)> = pixels
        .iter()
        .map(|c| (c.x, c.y, c.z))
        .collect();
    
    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::default(),
        Encoding::FAST_LOSSLESS,
        SpecificChannels::rgb(|pos: Vec2<usize>| {
            data[pos.y() * width as usize + pos.x()]
        }),
    );
    
    let image = Image::from_layer(layer);
    image.write().to_file(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tone_map_reinhard() {
        let color = Vec3::new(2.0, 2.0, 2.0);
        let mapped = ToneMap::Reinhard.apply(color);
        assert!(mapped.x < 1.0);
        assert!(mapped.x > 0.5);
    }

    #[test]
    fn test_tone_map_aces() {
        let color = Vec3::new(1.0, 1.0, 1.0);
        let mapped = ToneMap::Aces.apply(color);
        assert!(mapped.x <= 1.0);
        assert!(mapped.x >= 0.0);
    }

    #[test]
    fn test_exposure() {
        let color = Vec3::new(0.5, 0.5, 0.5);
        let exposed = apply_exposure(color, 1.0);
        assert!((exposed.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_srgb_roundtrip() {
        let linear = 0.5;
        let srgb = linear_to_srgb(linear);
        let back = srgb_to_linear(srgb);
        assert!((linear - back).abs() < 0.001);
    }

    #[test]
    fn test_float_to_rgbe() {
        let color = Vec3::new(1.0, 0.5, 0.25);
        let rgbe = float_to_rgbe(color);
        assert_eq!(rgbe[3], 129); // exp = 1
    }
}
