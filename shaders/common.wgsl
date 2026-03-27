// Common WGSL definitions for Nebula path tracer
// Note: WGSL doesn't support includes, so these definitions are duplicated in trace.wgsl
// This file serves as documentation and reference for the GPU data structures.

// =============================================================================
// Uniform Buffer Layout (binding 0)
// =============================================================================
// struct Uniforms {
//     width: u32,              // Image width
//     height: u32,             // Image height
//     samples: u32,            // Samples per pixel per frame
//     max_depth: u32,          // Maximum ray depth
//     camera_origin: vec4<f32>,
//     camera_lower_left: vec4<f32>,
//     camera_horizontal: vec4<f32>,
//     camera_vertical: vec4<f32>,
//     camera_u: vec4<f32>,
//     camera_v: vec4<f32>,
//     lens_radius: f32,
//     sphere_count: u32,
//     triangle_count: u32,
//     bvh_node_count: u32,
//     frame: u32,              // Current frame index (0-based)
//     total_frames: u32,       // Total frames for progressive rendering
//     accumulate: u32,         // 1 = accumulate to buffer, 0 = overwrite
//     _padding: f32,
// }

// =============================================================================
// Output Buffer Layout (binding 1)
// =============================================================================
// array<vec4<f32>> - Final output pixels (gamma-corrected)

// =============================================================================
// Sphere Buffer Layout (binding 2)
// =============================================================================
// struct Sphere {
//     center: vec4<f32>,       // xyz = center, w = radius
//     material: vec4<f32>,     // x = type, y = roughness/ior, z = emission, w = mat_id
//     albedo: vec4<f32>,       // rgb = color, a = unused
// }

// =============================================================================
// Triangle Buffer Layout (binding 3)
// =============================================================================
// struct Triangle {
//     v0: vec4<f32>,           // xyz = vertex 0, w = material_type
//     v1: vec4<f32>,           // xyz = vertex 1, w = roughness
//     v2: vec4<f32>,           // xyz = vertex 2, w = ior
//     n0: vec4<f32>,           // xyz = normal 0, w = emission
//     n1: vec4<f32>,           // xyz = normal 1, w = unused
//     n2: vec4<f32>,           // xyz = normal 2, w = unused
//     albedo: vec4<f32>,       // rgb = color, a = material_id
// }

// =============================================================================
// BVH Node Buffer Layout (binding 4)
// =============================================================================
// struct BVHNode {
//     min: vec4<f32>,          // xyz = AABB min, w = left_or_offset (bitcast to u32)
//     max: vec4<f32>,          // xyz = AABB max, w = count_axis_flags (bitcast to u32)
// }
//
// For interior nodes: count = 0, offset = right child index
// For leaf nodes: count > 0, offset = first primitive index
// Flags: count in upper 16 bits, axis in bits 8-15

// =============================================================================
// Material Buffer Layout (binding 5)
// =============================================================================
// struct GpuMaterial {
//     albedo: vec4<f32>,       // rgb = color, a = emission strength
//     params: vec4<f32>,       // x = type, y = roughness, z = ior, w = unused
// }

// =============================================================================
// Accumulation Buffer Layout (binding 6)
// =============================================================================
// array<vec4<f32>> - Running sum of samples for progressive rendering

// =============================================================================
// Material Types
// =============================================================================
// 0 = Lambertian (diffuse)
// 1 = Metal (specular with roughness)
// 2 = Dielectric (glass with refraction)
// 3 = Emissive (light source)

// =============================================================================
// Constants
// =============================================================================
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.001;
const MAX_STACK_SIZE: u32 = 64u;
