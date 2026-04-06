//! Image denoising for path traced renders
//!
//! Bilateral filter preserves edges while reducing noise from Monte Carlo sampling.

use crate::math::Vec3;

/// Denoising configuration
#[derive(Debug, Clone, Copy)]
pub struct DenoiseConfig {
    /// Spatial sigma (controls smoothing radius)
    pub sigma_spatial: f32,
    /// Range sigma (controls edge preservation)
    pub sigma_range: f32,
    /// Filter kernel radius (pixels)
    pub radius: i32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            sigma_spatial: 2.0,
            sigma_range: 0.1,
            radius: 5,
        }
    }
}

impl DenoiseConfig {
    /// Create config with given sigma values
    pub fn new(sigma_spatial: f32, sigma_range: f32) -> Self {
        let radius = (sigma_spatial * 2.5).ceil() as i32;
        Self {
            sigma_spatial,
            sigma_range,
            radius,
        }
    }

    /// Strong denoising (more blur, good for very noisy images)
    pub fn strong() -> Self {
        Self::new(4.0, 0.15)
    }

    /// Light denoising (preserve more detail)
    pub fn light() -> Self {
        Self::new(1.5, 0.05)
    }
}

/// Apply bilateral filter to denoise image
///
/// The bilateral filter smooths images while preserving edges by combining
/// spatial and range (color) weighting:
/// - Nearby pixels contribute more (spatial Gaussian)
/// - Similar-colored pixels contribute more (range Gaussian)
/// - Pixels across edges have different colors, so they're naturally excluded
pub fn bilateral_filter(
    pixels: &[Vec3],
    width: u32,
    height: u32,
    config: &DenoiseConfig,
) -> Vec<Vec3> {
    let mut output = vec![Vec3::zero(); pixels.len()];
    let w = width as i32;
    let h = height as i32;

    // Precompute spatial Gaussian weights
    let spatial_weights = precompute_spatial_weights(config.radius, config.sigma_spatial);
    let range_denom = 2.0 * config.sigma_range * config.sigma_range;

    for y in 0..h {
        for x in 0..w {
            let center_idx = (y * w + x) as usize;
            let center = pixels[center_idx];

            let mut sum = Vec3::zero();
            let mut weight_sum = 0.0;

            // Sample neighborhood
            for dy in -config.radius..=config.radius {
                let ny = y + dy;
                if ny < 0 || ny >= h {
                    continue;
                }

                for dx in -config.radius..=config.radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w {
                        continue;
                    }

                    let neighbor_idx = (ny * w + nx) as usize;
                    let neighbor = pixels[neighbor_idx];

                    // Spatial weight (precomputed)
                    let spatial_idx =
                        ((dy + config.radius) * (2 * config.radius + 1) + (dx + config.radius))
                            as usize;
                    let spatial_weight = spatial_weights[spatial_idx];

                    // Range weight (based on color difference)
                    let diff = center - neighbor;
                    let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    let range_weight = (-dist_sq / range_denom).exp();

                    // Combined weight
                    let weight = spatial_weight * range_weight;

                    sum = sum + neighbor * weight;
                    weight_sum += weight;
                }
            }

            // Normalize
            output[center_idx] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                center
            };
        }
    }

    output
}

/// Precompute spatial Gaussian weights for the kernel
fn precompute_spatial_weights(radius: i32, sigma: f32) -> Vec<f32> {
    let size = (2 * radius + 1) as usize;
    let mut weights = vec![0.0; size * size];
    let denom = 2.0 * sigma * sigma;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let dist_sq = (dx * dx + dy * dy) as f32;
            let weight = (-dist_sq / denom).exp();
            let idx = ((dy + radius) * (2 * radius + 1) + (dx + radius)) as usize;
            weights[idx] = weight;
        }
    }

    weights
}

/// Apply denoising with edge-aware strength
///
/// Adjusts denoising strength based on local variance (noise estimate).
/// High variance areas get more denoising, low variance areas preserved.
pub fn adaptive_bilateral(
    pixels: &[Vec3],
    width: u32,
    height: u32,
    base_config: &DenoiseConfig,
) -> Vec<Vec3> {
    let mut output = vec![Vec3::zero(); pixels.len()];
    let w = width as i32;
    let h = height as i32;

    // Estimate local variance at each pixel
    let variance = estimate_variance(pixels, width, height, 3);

    // Find variance range for normalization
    let max_var = variance.iter().cloned().fold(0.0_f32, f32::max);
    let max_var = max_var.max(0.001); // Avoid division by zero

    let spatial_weights = precompute_spatial_weights(base_config.radius, base_config.sigma_spatial);

    for y in 0..h {
        for x in 0..w {
            let center_idx = (y * w + x) as usize;
            let center = pixels[center_idx];

            // Adapt range sigma based on local variance
            // High variance = more noise = larger sigma (more smoothing)
            let local_var = variance[center_idx];
            let var_normalized = (local_var / max_var).sqrt();
            let adapted_sigma = base_config.sigma_range * (1.0 + var_normalized * 2.0);
            let range_denom = 2.0 * adapted_sigma * adapted_sigma;

            let mut sum = Vec3::zero();
            let mut weight_sum = 0.0;

            for dy in -base_config.radius..=base_config.radius {
                let ny = y + dy;
                if ny < 0 || ny >= h {
                    continue;
                }

                for dx in -base_config.radius..=base_config.radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w {
                        continue;
                    }

                    let neighbor_idx = (ny * w + nx) as usize;
                    let neighbor = pixels[neighbor_idx];

                    let spatial_idx = ((dy + base_config.radius) * (2 * base_config.radius + 1)
                        + (dx + base_config.radius)) as usize;
                    let spatial_weight = spatial_weights[spatial_idx];

                    let diff = center - neighbor;
                    let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    let range_weight = (-dist_sq / range_denom).exp();

                    let weight = spatial_weight * range_weight;
                    sum = sum + neighbor * weight;
                    weight_sum += weight;
                }
            }

            output[center_idx] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                center
            };
        }
    }

    output
}

/// Estimate local variance (noise) at each pixel
fn estimate_variance(pixels: &[Vec3], width: u32, height: u32, radius: i32) -> Vec<f32> {
    let mut variance = vec![0.0; pixels.len()];
    let w = width as i32;
    let h = height as i32;

    for y in 0..h {
        for x in 0..w {
            let center_idx = (y * w + x) as usize;
            let center = pixels[center_idx];

            let mut sum_sq_diff = 0.0;
            let mut count = 0;

            for dy in -radius..=radius {
                let ny = y + dy;
                if ny < 0 || ny >= h {
                    continue;
                }

                for dx in -radius..=radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w {
                        continue;
                    }

                    let neighbor_idx = (ny * w + nx) as usize;
                    let neighbor = pixels[neighbor_idx];
                    let diff = center - neighbor;
                    sum_sq_diff += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    count += 1;
                }
            }

            variance[center_idx] = if count > 0 {
                sum_sq_diff / count as f32
            } else {
                0.0
            };
        }
    }

    variance
}

/// Joint bilateral filter using auxiliary buffers (normals, albedo)
///
/// Guide images help preserve edges even better by using geometry information.
pub fn joint_bilateral(
    pixels: &[Vec3],
    normals: Option<&[Vec3]>,
    albedo: Option<&[Vec3]>,
    width: u32,
    height: u32,
    config: &DenoiseConfig,
) -> Vec<Vec3> {
    // If no guide images, fall back to regular bilateral
    if normals.is_none() && albedo.is_none() {
        return bilateral_filter(pixels, width, height, config);
    }

    let mut output = vec![Vec3::zero(); pixels.len()];
    let w = width as i32;
    let h = height as i32;

    let spatial_weights = precompute_spatial_weights(config.radius, config.sigma_spatial);
    let range_denom = 2.0 * config.sigma_range * config.sigma_range;
    let normal_denom = 0.5; // Sigma for normal difference
    let albedo_denom = 0.1; // Sigma for albedo difference

    for y in 0..h {
        for x in 0..w {
            let center_idx = (y * w + x) as usize;
            let center = pixels[center_idx];
            let center_normal = normals.map(|n| n[center_idx]);
            let center_albedo = albedo.map(|a| a[center_idx]);

            let mut sum = Vec3::zero();
            let mut weight_sum = 0.0;

            for dy in -config.radius..=config.radius {
                let ny = y + dy;
                if ny < 0 || ny >= h {
                    continue;
                }

                for dx in -config.radius..=config.radius {
                    let nx = x + dx;
                    if nx < 0 || nx >= w {
                        continue;
                    }

                    let neighbor_idx = (ny * w + nx) as usize;
                    let neighbor = pixels[neighbor_idx];

                    // Spatial weight
                    let spatial_idx = ((dy + config.radius) * (2 * config.radius + 1)
                        + (dx + config.radius)) as usize;
                    let spatial_weight = spatial_weights[spatial_idx];

                    // Color range weight
                    let color_diff = center - neighbor;
                    let color_dist_sq =
                        color_diff.x * color_diff.x + color_diff.y * color_diff.y + color_diff.z * color_diff.z;
                    let color_weight = (-color_dist_sq / range_denom).exp();

                    // Normal guide weight
                    let normal_weight = if let (Some(cn), Some(normals)) = (center_normal, normals) {
                        let nn = normals[neighbor_idx];
                        let dot = (cn.x * nn.x + cn.y * nn.y + cn.z * nn.z).clamp(-1.0, 1.0);
                        let angle = dot.acos();
                        (-angle * angle / normal_denom).exp()
                    } else {
                        1.0
                    };

                    // Albedo guide weight
                    let albedo_weight = if let (Some(ca), Some(albedo)) = (center_albedo, albedo) {
                        let na = albedo[neighbor_idx];
                        let diff = ca - na;
                        let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        (-dist_sq / albedo_denom).exp()
                    } else {
                        1.0
                    };

                    let weight = spatial_weight * color_weight * normal_weight * albedo_weight;
                    sum = sum + neighbor * weight;
                    weight_sum += weight;
                }
            }

            output[center_idx] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                center
            };
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_config_default() {
        let config = DenoiseConfig::default();
        assert_eq!(config.sigma_spatial, 2.0);
        assert_eq!(config.sigma_range, 0.1);
        assert_eq!(config.radius, 5);
    }

    #[test]
    fn test_denoise_config_new() {
        let config = DenoiseConfig::new(3.0, 0.2);
        assert_eq!(config.sigma_spatial, 3.0);
        assert_eq!(config.sigma_range, 0.2);
        assert_eq!(config.radius, 8); // ceil(3.0 * 2.5)
    }

    #[test]
    fn test_precompute_spatial_weights() {
        let weights = precompute_spatial_weights(1, 1.0);
        assert_eq!(weights.len(), 9); // 3x3
        // Center should have weight 1.0
        assert!((weights[4] - 1.0).abs() < 0.001);
        // Corners should have lower weight
        assert!(weights[0] < weights[4]);
    }

    #[test]
    fn test_bilateral_preserves_uniform() {
        // Uniform image should stay uniform
        let pixels = vec![Vec3::new(0.5, 0.5, 0.5); 9];
        let config = DenoiseConfig::new(1.0, 0.1);
        let result = bilateral_filter(&pixels, 3, 3, &config);

        for pixel in &result {
            assert!((pixel.x - 0.5).abs() < 0.01);
            assert!((pixel.y - 0.5).abs() < 0.01);
            assert!((pixel.z - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_bilateral_preserves_edges() {
        // Sharp edge: left half black, right half white
        let mut pixels = vec![Vec3::zero(); 16];
        for y in 0..4 {
            for x in 2..4 {
                pixels[y * 4 + x] = Vec3::new(1.0, 1.0, 1.0);
            }
        }

        let config = DenoiseConfig::new(1.0, 0.05);
        let result = bilateral_filter(&pixels, 4, 4, &config);

        // Center of black region should stay dark
        assert!(result[5].x < 0.2); // (1,1)
        // Center of white region should stay bright
        assert!(result[6].x > 0.8); // (2,1)
    }

    #[test]
    fn test_bilateral_reduces_noise() {
        // Create noisy image
        let mut pixels = vec![Vec3::new(0.5, 0.5, 0.5); 25];
        // Add noise to some pixels
        pixels[6] = Vec3::new(0.6, 0.5, 0.5);
        pixels[8] = Vec3::new(0.4, 0.5, 0.5);
        pixels[12] = Vec3::new(0.55, 0.45, 0.5);

        let config = DenoiseConfig::new(1.5, 0.2);
        let result = bilateral_filter(&pixels, 5, 5, &config);

        // Noisy pixels should be closer to 0.5
        let center_before = (pixels[12].x - 0.5).abs();
        let center_after = (result[12].x - 0.5).abs();
        assert!(center_after < center_before);
    }

    #[test]
    fn test_estimate_variance() {
        // Uniform region has zero variance
        let uniform = vec![Vec3::new(0.5, 0.5, 0.5); 9];
        let variance = estimate_variance(&uniform, 3, 3, 1);
        assert!(variance[4] < 0.001);

        // Noisy region has higher variance
        let mut noisy = vec![Vec3::new(0.5, 0.5, 0.5); 9];
        noisy[0] = Vec3::new(0.3, 0.3, 0.3);
        noisy[2] = Vec3::new(0.7, 0.7, 0.7);
        let variance = estimate_variance(&noisy, 3, 3, 1);
        assert!(variance[1] > 0.01);
    }

    #[test]
    fn test_adaptive_bilateral() {
        // Should work without crashing
        let pixels = vec![Vec3::new(0.5, 0.5, 0.5); 25];
        let config = DenoiseConfig::default();
        let result = adaptive_bilateral(&pixels, 5, 5, &config);
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_joint_bilateral_fallback() {
        // Without guides, should produce same result as bilateral
        let pixels = vec![Vec3::new(0.5, 0.5, 0.5); 9];
        let config = DenoiseConfig::new(1.0, 0.1);

        let result1 = bilateral_filter(&pixels, 3, 3, &config);
        let result2 = joint_bilateral(&pixels, None, None, 3, 3, &config);

        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((a.x - b.x).abs() < 0.001);
        }
    }

    #[test]
    fn test_joint_bilateral_with_normals() {
        let pixels = vec![Vec3::new(0.5, 0.5, 0.5); 9];
        let normals = vec![Vec3::new(0.0, 0.0, 1.0); 9];
        let config = DenoiseConfig::new(1.0, 0.1);

        let result = joint_bilateral(&pixels, Some(&normals), None, 3, 3, &config);
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_strong_vs_light_config() {
        let strong = DenoiseConfig::strong();
        let light = DenoiseConfig::light();

        assert!(strong.sigma_spatial > light.sigma_spatial);
        assert!(strong.sigma_range > light.sigma_range);
        assert!(strong.radius > light.radius);
    }
}
