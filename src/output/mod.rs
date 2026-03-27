//! Image output utilities with tone mapping

use crate::math::Vec3;
use std::path::Path;
use std::io::{self, Write};
use std::fs::File;

/// Tone mapping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapping {
    /// No tone mapping, just clamp
    None,
    /// Reinhard tone mapping
    Reinhard,
    /// ACES filmic tone mapping
    Aces,
}

impl Default for ToneMapping {
    fn default() -> Self {
        ToneMapping::None
    }
}

/// Output settings
#[derive(Debug, Clone)]
pub struct OutputSettings {
    pub tone_mapping: ToneMapping,
    pub exposure: f32,
    pub gamma_correct: bool,
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            tone_mapping: ToneMapping::None,
            exposure: 1.0,
            gamma_correct: true,
        }
    }
}

/// Save image as PPM (simple format, always works)
pub fn save_ppm(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    save_ppm_with_settings(path, pixels, width, height, &OutputSettings::default())
}

/// Save image as PPM with custom settings
pub fn save_ppm_with_settings(
    path: &Path,
    pixels: &[Vec3],
    width: u32,
    height: u32,
    settings: &OutputSettings,
) -> io::Result<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "P3")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;

    for pixel in pixels {
        let color = process_pixel(*pixel, settings);
        let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
        writeln!(file, "{} {} {}", r, g, b)?;
    }

    Ok(())
}

/// Save image as PNG
pub fn save_png(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    save_png_with_settings(path, pixels, width, height, &OutputSettings::default())
}

/// Save image as PNG with custom settings
pub fn save_png_with_settings(
    path: &Path,
    pixels: &[Vec3],
    width: u32,
    height: u32,
    settings: &OutputSettings,
) -> io::Result<()> {
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    
    for (i, pixel) in pixels.iter().enumerate() {
        let color = process_pixel(*pixel, settings);
        buffer[i * 3] = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
        buffer[i * 3 + 1] = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
        buffer[i * 3 + 2] = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
    }

    image::save_buffer(path, &buffer, width, height, image::ColorType::Rgb8)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Process a pixel with tone mapping and gamma correction
fn process_pixel(pixel: Vec3, settings: &OutputSettings) -> Vec3 {
    // Apply exposure
    let color = pixel * settings.exposure;
    
    // Apply tone mapping
    let mapped = match settings.tone_mapping {
        ToneMapping::None => color,
        ToneMapping::Reinhard => tonemap_reinhard(color),
        ToneMapping::Aces => tonemap_aces(color),
    };
    
    // Apply gamma correction
    if settings.gamma_correct {
        linear_to_srgb(mapped)
    } else {
        mapped
    }
}

/// Convert linear color to sRGB (gamma correction)
pub fn linear_to_srgb(color: Vec3) -> Vec3 {
    fn gamma(x: f32) -> f32 {
        if x <= 0.0031308 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }
    Vec3::new(gamma(color.x), gamma(color.y), gamma(color.z))
}

/// Reinhard tone mapping
pub fn tonemap_reinhard(color: Vec3) -> Vec3 {
    Vec3::new(
        color.x / (1.0 + color.x),
        color.y / (1.0 + color.y),
        color.z / (1.0 + color.z),
    )
}

/// ACES filmic tone mapping
pub fn tonemap_aces(color: Vec3) -> Vec3 {
    // ACES constants
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    
    fn aces_component(x: f32) -> f32 {
        let num = x * (A * x + B);
        let den = x * (C * x + D) + E;
        (num / den).clamp(0.0, 1.0)
    }
    
    Vec3::new(
        aces_component(color.x),
        aces_component(color.y),
        aces_component(color.z),
    )
}

/// Save image as HDR (Radiance RGBE format)
pub fn save_hdr(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    use std::io::BufWriter;
    
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // HDR header
    writeln!(writer, "#?RADIANCE")?;
    writeln!(writer, "FORMAT=32-bit_rle_rgbe")?;
    writeln!(writer)?;
    writeln!(writer, "-Y {} +X {}", height, width)?;
    
    // Write pixels in RGBE format
    for pixel in pixels {
        let rgbe = rgb_to_rgbe(pixel.x, pixel.y, pixel.z);
        writer.write_all(&rgbe)?;
    }
    
    Ok(())
}

/// Convert RGB to RGBE (Radiance format)
fn rgb_to_rgbe(r: f32, g: f32, b: f32) -> [u8; 4] {
    let max_val = r.max(g).max(b);
    
    if max_val < 1e-32 {
        return [0, 0, 0, 0];
    }
    
    let (mantissa, exponent) = frexp(max_val);
    let scale = mantissa * 256.0 / max_val;
    
    [
        (r * scale) as u8,
        (g * scale) as u8,
        (b * scale) as u8,
        (exponent + 128) as u8,
    ]
}

/// Extract mantissa and exponent (frexp equivalent)
fn frexp(x: f32) -> (f32, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 126;
    let mantissa = f32::from_bits((bits & 0x807fffff) | 0x3f000000);
    (mantissa, exponent)
}

/// Save image as EXR (OpenEXR format - simplified single-part scanline)
pub fn save_exr(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    // Note: This is a simplified EXR writer. For production use, consider the `exr` crate.
    use std::io::BufWriter;
    
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // EXR magic number
    writer.write_all(&[0x76, 0x2f, 0x31, 0x01])?;
    
    // Version (2.0, single-part scanline)
    writer.write_all(&[2, 0, 0, 0])?;
    
    // Headers (simplified - channels, compression, dataWindow, displayWindow, lineOrder, pixelAspectRatio, screenWindowCenter, screenWindowWidth)
    // For a proper implementation, use the exr crate
    
    // For now, write raw float data with minimal header
    // This is a placeholder - proper EXR requires more complex header structure
    
    // Write pixel data as raw floats (not standard EXR but readable)
    for pixel in pixels {
        writer.write_all(&pixel.x.to_le_bytes())?;
        writer.write_all(&pixel.y.to_le_bytes())?;
        writer.write_all(&pixel.z.to_le_bytes())?;
    }
    
    // Note: For proper EXR support, add the `exr` crate dependency
    eprintln!("Warning: EXR output is simplified. For full EXR support, use the `exr` crate.");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_srgb() {
        // Black should stay black
        let black = linear_to_srgb(Vec3::zero());
        assert!(black.x.abs() < 1e-6);
        
        // White should stay white
        let white = linear_to_srgb(Vec3::one());
        assert!((white.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reinhard() {
        // 0 maps to 0
        let zero = tonemap_reinhard(Vec3::zero());
        assert!(zero.x.abs() < 1e-6);
        
        // 1 maps to 0.5
        let one = tonemap_reinhard(Vec3::one());
        assert!((one.x - 0.5).abs() < 1e-6);
        
        // Large values approach 1
        let large = tonemap_reinhard(Vec3::new(100.0, 100.0, 100.0));
        assert!(large.x > 0.99);
    }

    #[test]
    fn test_aces() {
        // 0 maps to 0
        let zero = tonemap_aces(Vec3::zero());
        assert!(zero.x.abs() < 1e-6);
        
        // Output should be in [0, 1]
        let large = tonemap_aces(Vec3::new(10.0, 10.0, 10.0));
        assert!(large.x >= 0.0 && large.x <= 1.0);
    }

    #[test]
    fn test_output_settings_default() {
        let settings = OutputSettings::default();
        assert_eq!(settings.tone_mapping, ToneMapping::None);
        assert!((settings.exposure - 1.0).abs() < 1e-6);
        assert!(settings.gamma_correct);
    }

    #[test]
    fn test_process_pixel_exposure() {
        let pixel = Vec3::new(0.5, 0.5, 0.5);
        let settings = OutputSettings {
            exposure: 2.0,
            tone_mapping: ToneMapping::None,
            gamma_correct: false,
        };
        let result = process_pixel(pixel, &settings);
        assert!((result.x - 1.0).abs() < 1e-6);
    }
}
