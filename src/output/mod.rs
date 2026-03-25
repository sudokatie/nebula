//! Image output utilities

use crate::math::Vec3;
use std::path::Path;
use std::io::{self, Write};
use std::fs::File;

/// Save image as PPM (simple format, always works)
pub fn save_ppm(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    let mut file = File::create(path)?;
    
    writeln!(file, "P3")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;

    for pixel in pixels {
        let color = linear_to_srgb(*pixel);
        let r = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
        writeln!(file, "{} {} {}", r, g, b)?;
    }

    Ok(())
}

/// Save image as PNG
pub fn save_png(path: &Path, pixels: &[Vec3], width: u32, height: u32) -> io::Result<()> {
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    
    for (i, pixel) in pixels.iter().enumerate() {
        let color = linear_to_srgb(*pixel);
        buffer[i * 3] = (color.x.clamp(0.0, 1.0) * 255.0) as u8;
        buffer[i * 3 + 1] = (color.y.clamp(0.0, 1.0) * 255.0) as u8;
        buffer[i * 3 + 2] = (color.z.clamp(0.0, 1.0) * 255.0) as u8;
    }

    image::save_buffer(path, &buffer, width, height, image::ColorType::Rgb8)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

/// Convert linear color to sRGB
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

/// ACES filmic tone mapping
#[allow(dead_code)]
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
