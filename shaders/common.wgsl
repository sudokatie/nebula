// Common shader utilities for Nebula path tracer

// GPU limits
const MAX_SPHERES: u32 = 1024u;
const MAX_TRIANGLES: u32 = 65536u;
const MAX_BVH_NODES: u32 = 131072u;
const MAX_MATERIALS: u32 = 256u;

// Material types
const MAT_LAMBERTIAN: u32 = 0u;
const MAT_METAL: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

// Sphere data structure
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_id: u32,
    _padding: vec3<u32>,
}

// Triangle data structure
struct Triangle {
    v0: vec3<f32>,
    _pad0: f32,
    v1: vec3<f32>,
    _pad1: f32,
    v2: vec3<f32>,
    _pad2: f32,
    n0: vec3<f32>,
    _pad3: f32,
    n1: vec3<f32>,
    _pad4: f32,
    n2: vec3<f32>,
    material_id: u32,
}

// Material data structure
struct Material {
    albedo: vec3<f32>,
    material_type: u32,
    roughness: f32,
    ior: f32,
    emission_strength: f32,
    _padding: f32,
}

// BVH node structure
struct BVHNode {
    min: vec3<f32>,
    left_or_prim: i32,  // If leaf: primitive index, else: left child
    max: vec3<f32>,
    right_or_count: i32, // If leaf: primitive count, else: right child
}

// AABB intersection test
fn hit_aabb(
    box_min: vec3<f32>,
    box_max: vec3<f32>,
    ray_origin: vec3<f32>,
    ray_inv_dir: vec3<f32>,
    t_min: f32,
    t_max: f32
) -> bool {
    let t0 = (box_min - ray_origin) * ray_inv_dir;
    let t1 = (box_max - ray_origin) * ray_inv_dir;
    
    let t_near = min(t0, t1);
    let t_far = max(t0, t1);
    
    let t_enter = max(max(t_near.x, t_near.y), max(t_near.z, t_min));
    let t_exit = min(min(t_far.x, t_far.y), min(t_far.z, t_max));
    
    return t_enter <= t_exit;
}

// Triangle intersection (Moller-Trumbore)
fn hit_triangle(
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    t_min: f32,
    t_max: f32
) -> vec4<f32> {
    // Returns vec4(t, u, v, hit) where hit > 0 means intersection
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_direction, edge2);
    let a = dot(edge1, h);
    
    if abs(a) < 1e-8 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    
    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);
    
    if u < 0.0 || u > 1.0 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    
    let q = cross(s, edge1);
    let v = f * dot(ray_direction, q);
    
    if v < 0.0 || u + v > 1.0 {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    
    let t = f * dot(edge2, q);
    
    if t < t_min || t > t_max {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    
    return vec4<f32>(t, u, v, 1.0);
}

// Sphere intersection
fn hit_sphere_common(
    center: vec3<f32>,
    radius: f32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    t_min: f32,
    t_max: f32
) -> vec2<f32> {
    // Returns vec2(t, hit) where hit > 0 means intersection
    let oc = ray_origin - center;
    let a = dot(ray_direction, ray_direction);
    let half_b = dot(oc, ray_direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    
    if discriminant < 0.0 {
        return vec2<f32>(-1.0, 0.0);
    }
    
    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    
    if root <= t_min || root >= t_max {
        root = (-half_b + sqrtd) / a;
        if root <= t_min || root >= t_max {
            return vec2<f32>(-1.0, 0.0);
        }
    }
    
    return vec2<f32>(root, 1.0);
}

// Schlick approximation for Fresnel
fn schlick(cosine: f32, ior: f32) -> f32 {
    var r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// GGX normal distribution
fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_sq = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / (3.14159265359 * denom * denom);
}

// Smith geometry function
fn smith_g1(n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha_sq = alpha * alpha;
    let n_dot_v_sq = n_dot_v * n_dot_v;
    return 2.0 * n_dot_v / (n_dot_v + sqrt(alpha_sq + (1.0 - alpha_sq) * n_dot_v_sq));
}
