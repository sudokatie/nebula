// Path tracing compute shader with BVH traversal, proper primitive indexing, and texture support

struct Uniforms {
    width: u32,
    height: u32,
    samples: u32,
    max_depth: u32,
    camera_origin: vec4<f32>,
    camera_lower_left: vec4<f32>,
    camera_horizontal: vec4<f32>,
    camera_vertical: vec4<f32>,
    camera_u: vec4<f32>,
    camera_v: vec4<f32>,
    lens_radius: f32,
    sphere_count: u32,
    triangle_count: u32,
    bvh_node_count: u32,
    frame: u32,
    total_frames: u32,
    accumulate: u32,
    texture_count: u32,
}

struct Sphere {
    center: vec4<f32>,      // xyz = center, w = radius
    material: vec4<f32>,    // x = material_type, y = roughness/ior, z = emission_strength, w = mat_id
    albedo: vec4<f32>,      // rgb = color, a = texture_id
}

struct Triangle {
    v0: vec4<f32>,          // xyz = vertex 0, w = material_type
    v1: vec4<f32>,          // xyz = vertex 1, w = roughness
    v2: vec4<f32>,          // xyz = vertex 2, w = ior
    n0: vec4<f32>,          // xyz = normal 0, w = emission
    n1: vec4<f32>,          // xyz = normal 1, w = u0
    n2: vec4<f32>,          // xyz = normal 2, w = v0
    albedo: vec4<f32>,      // rgb = color, a = material_id
    uvs: vec4<f32>,         // u1, v1, u2, v2
}

struct BVHNode {
    min: vec4<f32>,         // xyz = AABB min, w = left_or_offset (as bits)
    max: vec4<f32>,         // xyz = AABB max, w = count_axis_flags (as bits)
}

struct GpuMaterial {
    albedo: vec4<f32>,      // rgb = color, a = emission
    params: vec4<f32>,      // x = type, y = roughness, z = ior, w = texture_id
    checker: vec4<f32>,     // checker color 1 (rgb) + scale (a)
    checker2: vec4<f32>,    // checker color 2 (rgb) + is_checker flag (a)
}

struct PrimitiveRef {
    prim_type: u32,         // 0 = sphere, 1 = triangle
    index: u32,
    _pad0: u32,
    _pad1: u32,
}

struct TextureInfo {
    width: u32,
    height: u32,
    offset: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> output: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> spheres: array<Sphere>;

@group(0) @binding(3)
var<storage, read> triangles: array<Triangle>;

@group(0) @binding(4)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(0) @binding(5)
var<storage, read> materials: array<GpuMaterial>;

@group(0) @binding(6)
var<storage, read_write> accumulation: array<vec4<f32>>;

@group(0) @binding(7)
var<storage, read> primitive_refs: array<PrimitiveRef>;

@group(0) @binding(8)
var<storage, read> texture_infos: array<TextureInfo>;

@group(0) @binding(9)
var<storage, read> texture_data: array<f32>;

// Constants
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.001;
const MAX_STACK_SIZE: u32 = 64u;

// PCG random number generator
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) / 4294967295.0;
}

fn rand_in_unit_sphere(seed: ptr<function, u32>) -> vec3<f32> {
    loop {
        let p = vec3<f32>(
            rand(seed) * 2.0 - 1.0,
            rand(seed) * 2.0 - 1.0,
            rand(seed) * 2.0 - 1.0
        );
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

fn rand_unit_vector(seed: ptr<function, u32>) -> vec3<f32> {
    return normalize(rand_in_unit_sphere(seed));
}

fn rand_in_unit_disk(seed: ptr<function, u32>) -> vec2<f32> {
    loop {
        let p = vec2<f32>(rand(seed) * 2.0 - 1.0, rand(seed) * 2.0 - 1.0);
        if dot(p, p) < 1.0 {
            return p;
        }
    }
}

fn rand_cosine_direction(seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand(seed);
    let r2 = rand(seed);
    let z = sqrt(1.0 - r2);
    let phi = 2.0 * PI * r1;
    let x = cos(phi) * sqrt(r2);
    let y = sin(phi) * sqrt(r2);
    return vec3<f32>(x, y, z);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.direction * t;
}

fn get_camera_ray(u: f32, v: f32, seed: ptr<function, u32>) -> Ray {
    let rd = rand_in_unit_disk(seed) * uniforms.lens_radius;
    let offset = uniforms.camera_u.xyz * rd.x + uniforms.camera_v.xyz * rd.y;
    let origin = uniforms.camera_origin.xyz + offset;
    let direction = uniforms.camera_lower_left.xyz 
        + uniforms.camera_horizontal.xyz * u 
        + uniforms.camera_vertical.xyz * v 
        - origin;
    return Ray(origin, normalize(direction));
}

struct Material {
    albedo: vec3<f32>,
    material_type: u32,     // 0=lambertian, 1=metal, 2=dielectric, 3=emissive
    roughness: f32,
    ior: f32,
    emission: f32,
}

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    material: Material,
    uv: vec2<f32>,
}

// Sample checker texture
fn sample_checker(point: vec3<f32>, color1: vec3<f32>, color2: vec3<f32>, scale: f32) -> vec3<f32> {
    let sines = sin(scale * point.x) * sin(scale * point.y) * sin(scale * point.z);
    if sines < 0.0 {
        return color1;
    }
    return color2;
}

// Sample image texture
fn sample_texture(tex_id: u32, uv: vec2<f32>) -> vec3<f32> {
    if tex_id >= uniforms.texture_count {
        return vec3<f32>(1.0, 0.0, 1.0); // Magenta for missing texture
    }
    
    let info = texture_infos[tex_id];
    if info.width == 0u || info.height == 0u {
        return vec3<f32>(0.5, 0.5, 0.5);
    }
    
    let u_clamped = clamp(uv.x, 0.0, 1.0);
    let v_clamped = 1.0 - clamp(uv.y, 0.0, 1.0); // Flip V
    
    let i = u32(u_clamped * f32(info.width - 1u));
    let j = u32(v_clamped * f32(info.height - 1u));
    
    let pixel_idx = info.offset + (j * info.width + i) * 3u;
    
    return vec3<f32>(
        texture_data[pixel_idx],
        texture_data[pixel_idx + 1u],
        texture_data[pixel_idx + 2u]
    );
}

// Sample procedural noise
fn noise_hash(p: vec3<i32>) -> f32 {
    let n = p.x + p.y * 57 + p.z * 131;
    let n2 = (n << 13) ^ n;
    let m = n2 * (n2 * n2 * 15731 + 789221) + 1376312589;
    return 1.0 - f32(m & 0x7fffffff) / 1073741824.0;
}

fn sample_noise(point: vec3<f32>, scale: f32) -> f32 {
    let p = point * scale;
    let pi = vec3<i32>(i32(floor(p.x)), i32(floor(p.y)), i32(floor(p.z)));
    let pf = fract(p);
    
    let ux = pf.x * pf.x * (3.0 - 2.0 * pf.x);
    let uy = pf.y * pf.y * (3.0 - 2.0 * pf.y);
    let uz = pf.z * pf.z * (3.0 - 2.0 * pf.z);
    
    let n000 = noise_hash(pi);
    let n001 = noise_hash(pi + vec3<i32>(0, 0, 1));
    let n010 = noise_hash(pi + vec3<i32>(0, 1, 0));
    let n011 = noise_hash(pi + vec3<i32>(0, 1, 1));
    let n100 = noise_hash(pi + vec3<i32>(1, 0, 0));
    let n101 = noise_hash(pi + vec3<i32>(1, 0, 1));
    let n110 = noise_hash(pi + vec3<i32>(1, 1, 0));
    let n111 = noise_hash(pi + vec3<i32>(1, 1, 1));
    
    let nx00 = mix(n000, n100, ux);
    let nx01 = mix(n001, n101, ux);
    let nx10 = mix(n010, n110, ux);
    let nx11 = mix(n011, n111, ux);
    
    let nxy0 = mix(nx00, nx10, uy);
    let nxy1 = mix(nx01, nx11, uy);
    
    return mix(nxy0, nxy1, uz) * 0.5 + 0.5;
}

// AABB intersection test
fn hit_aabb(node: BVHNode, ray_origin: vec3<f32>, inv_dir: vec3<f32>, t_min: f32, t_max: f32) -> bool {
    let aabb_min = node.min.xyz;
    let aabb_max = node.max.xyz;
    
    let t0 = (aabb_min - ray_origin) * inv_dir;
    let t1 = (aabb_max - ray_origin) * inv_dir;
    
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    
    let tmin_val = max(max(tmin3.x, tmin3.y), max(tmin3.z, t_min));
    let tmax_val = min(min(tmax3.x, tmax3.y), min(tmax3.z, t_max));
    
    return tmax_val >= tmin_val;
}

fn hit_sphere(sphere: Sphere, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    let center = sphere.center.xyz;
    let radius = sphere.center.w;
    
    let oc = ray.origin - center;
    let a = dot(ray.direction, ray.direction);
    let half_b = dot(oc, ray.direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    
    var rec: HitRecord;
    rec.t = -1.0;
    
    if discriminant < 0.0 {
        return rec;
    }
    
    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    
    if root <= t_min || root >= t_max {
        root = (-half_b + sqrtd) / a;
        if root <= t_min || root >= t_max {
            return rec;
        }
    }
    
    rec.t = root;
    rec.point = ray_at(ray, root);
    let outward_normal = (rec.point - center) / radius;
    rec.front_face = dot(ray.direction, outward_normal) < 0.0;
    rec.normal = select(-outward_normal, outward_normal, rec.front_face);
    
    // Compute sphere UV
    let theta = acos(-outward_normal.y);
    let phi = atan2(-outward_normal.z, outward_normal.x) + PI;
    rec.uv = vec2<f32>(phi / (2.0 * PI), theta / PI);
    
    // Extract material
    rec.material.albedo = sphere.albedo.xyz;
    rec.material.material_type = u32(sphere.material.x);
    rec.material.roughness = sphere.material.y;
    rec.material.ior = sphere.material.y;
    rec.material.emission = sphere.material.z;
    
    return rec;
}

// Moller-Trumbore triangle intersection with UV interpolation
fn hit_triangle(tri: Triangle, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var rec: HitRecord;
    rec.t = -1.0;
    
    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;
    
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    
    if abs(a) < 1e-8 {
        return rec;
    }
    
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    
    if u < 0.0 || u > 1.0 {
        return rec;
    }
    
    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    
    if v < 0.0 || u + v > 1.0 {
        return rec;
    }
    
    let t = f * dot(edge2, q);
    
    if t <= t_min || t >= t_max {
        return rec;
    }
    
    rec.t = t;
    rec.point = ray_at(ray, t);
    
    // Interpolate normal
    let w = 1.0 - u - v;
    let outward_normal = normalize(tri.n0.xyz * w + tri.n1.xyz * u + tri.n2.xyz * v);
    rec.front_face = dot(ray.direction, outward_normal) < 0.0;
    rec.normal = select(-outward_normal, outward_normal, rec.front_face);
    
    // Interpolate UVs
    let uv0 = vec2<f32>(tri.n1.w, tri.n2.w);
    let uv1 = vec2<f32>(tri.uvs.x, tri.uvs.y);
    let uv2 = vec2<f32>(tri.uvs.z, tri.uvs.w);
    rec.uv = uv0 * w + uv1 * u + uv2 * v;
    
    // Extract material
    rec.material.albedo = tri.albedo.xyz;
    rec.material.material_type = u32(tri.v0.w);
    rec.material.roughness = tri.v1.w;
    rec.material.ior = tri.v2.w;
    rec.material.emission = tri.n0.w;
    
    return rec;
}

// Hit a primitive by reference
fn hit_primitive(prim_ref: PrimitiveRef, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    if prim_ref.prim_type == 0u {
        // Sphere
        if prim_ref.index < uniforms.sphere_count {
            return hit_sphere(spheres[prim_ref.index], ray, t_min, t_max);
        }
    } else {
        // Triangle
        if prim_ref.index < uniforms.triangle_count {
            return hit_triangle(triangles[prim_ref.index], ray, t_min, t_max);
        }
    }
    
    var rec: HitRecord;
    rec.t = -1.0;
    return rec;
}

// BVH traversal with proper primitive indirection
fn hit_scene_bvh(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest: HitRecord;
    closest.t = -1.0;
    var closest_t = t_max;
    
    if uniforms.bvh_node_count == 0u {
        return hit_scene_linear(ray, t_min, t_max);
    }
    
    let inv_dir = vec3<f32>(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
    let dir_is_neg = vec3<bool>(ray.direction.x < 0.0, ray.direction.y < 0.0, ray.direction.z < 0.0);
    
    var stack: array<u32, 64>;
    var stack_ptr = 0u;
    var current = 0u;
    
    loop {
        if current >= uniforms.bvh_node_count {
            if stack_ptr == 0u {
                break;
            }
            stack_ptr -= 1u;
            current = stack[stack_ptr];
            continue;
        }
        
        let node = bvh_nodes[current];
        
        if hit_aabb(node, ray.origin, inv_dir, t_min, closest_t) {
            let offset_bits = bitcast<u32>(node.min.w);
            let flags_bits = bitcast<u32>(node.max.w);
            let count = (flags_bits >> 16u) & 0xFFFFu;
            let axis = (flags_bits >> 8u) & 0xFFu;
            
            if count > 0u {
                // Leaf node - test primitives using indirection
                let prim_start = offset_bits;
                for (var i = 0u; i < count; i++) {
                    let prim_idx = prim_start + i;
                    let prim_ref = primitive_refs[prim_idx];
                    let hit = hit_primitive(prim_ref, ray, t_min, closest_t);
                    if hit.t > 0.0 {
                        closest = hit;
                        closest_t = hit.t;
                    }
                }
                
                if stack_ptr == 0u {
                    break;
                }
                stack_ptr -= 1u;
                current = stack[stack_ptr];
            } else {
                // Interior node
                let left = current + 1u;
                let right = offset_bits;
                
                if select(false, true, dir_is_neg[axis]) {
                    stack[stack_ptr] = left;
                    stack_ptr += 1u;
                    current = right;
                } else {
                    stack[stack_ptr] = right;
                    stack_ptr += 1u;
                    current = left;
                }
            }
        } else {
            if stack_ptr == 0u {
                break;
            }
            stack_ptr -= 1u;
            current = stack[stack_ptr];
        }
    }
    
    return closest;
}

// Linear scene intersection (fallback)
fn hit_scene_linear(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest: HitRecord;
    closest.t = -1.0;
    var closest_t = t_max;
    
    for (var i = 0u; i < uniforms.sphere_count; i++) {
        let hit = hit_sphere(spheres[i], ray, t_min, closest_t);
        if hit.t > 0.0 {
            closest = hit;
            closest_t = hit.t;
        }
    }
    
    for (var i = 0u; i < uniforms.triangle_count; i++) {
        let hit = hit_triangle(triangles[i], ray, t_min, closest_t);
        if hit.t > 0.0 {
            closest = hit;
            closest_t = hit.t;
        }
    }
    
    return closest;
}

fn hit_scene(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    if uniforms.bvh_node_count > 0u {
        return hit_scene_bvh(ray, t_min, t_max);
    }
    return hit_scene_linear(ray, t_min, t_max);
}

// Build orthonormal basis
fn build_onb(n: vec3<f32>) -> mat3x3<f32> {
    let w = n;
    var a: vec3<f32>;
    if abs(w.x) > 0.9 {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let v = normalize(cross(w, a));
    let u = cross(w, v);
    return mat3x3<f32>(u, v, w);
}

fn reflect_vec(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract_vec(uv: vec3<f32>, n: vec3<f32>, etai_over_etat: f32) -> vec3<f32> {
    let cos_theta = min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

fn fresnel_schlick(cos_theta: f32, ior: f32) -> f32 {
    var r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

struct ScatterResult {
    scattered: Ray,
    attenuation: vec3<f32>,
    did_scatter: bool,
}

fn scatter(ray: Ray, hit: HitRecord, seed: ptr<function, u32>) -> ScatterResult {
    var result: ScatterResult;
    result.did_scatter = false;
    
    // Get material and potentially apply textures
    var albedo = hit.material.albedo;
    let mat_id = u32(hit.material.roughness); // Actually stored in material params
    
    // Check for checker pattern on lambertian materials
    if hit.material.material_type == 0u && mat_id < 256u {
        let mat = materials[mat_id];
        if mat.checker2.w > 0.5 {
            // This is a checker material
            let c1 = mat.checker.xyz;
            let c2 = mat.checker2.xyz;
            let scale = mat.checker.w;
            albedo = sample_checker(hit.point, c1, c2, scale);
        }
        
        // Check for texture
        let tex_id = i32(mat.params.w);
        if tex_id >= 0 {
            albedo = sample_texture(u32(tex_id), hit.uv);
        }
    }
    
    switch hit.material.material_type {
        case 0u: {
            // Lambertian
            let local_dir = rand_cosine_direction(seed);
            let onb = build_onb(hit.normal);
            let scatter_dir = onb * local_dir;
            result.scattered = Ray(hit.point, normalize(scatter_dir));
            result.attenuation = albedo;
            result.did_scatter = true;
        }
        case 1u: {
            // Metal
            let reflected = reflect_vec(normalize(ray.direction), hit.normal);
            let fuzz = rand_in_unit_sphere(seed) * hit.material.roughness;
            result.scattered = Ray(hit.point, normalize(reflected + fuzz));
            result.attenuation = albedo;
            result.did_scatter = dot(result.scattered.direction, hit.normal) > 0.0;
        }
        case 2u: {
            // Dielectric
            result.attenuation = vec3<f32>(1.0);
            let refraction_ratio = select(hit.material.ior, 1.0 / hit.material.ior, hit.front_face);
            let unit_dir = normalize(ray.direction);
            let cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            
            let cannot_refract = refraction_ratio * sin_theta > 1.0;
            let reflectance = fresnel_schlick(cos_theta, refraction_ratio);
            
            var direction: vec3<f32>;
            if cannot_refract || reflectance > rand(seed) {
                direction = reflect_vec(unit_dir, hit.normal);
            } else {
                direction = refract_vec(unit_dir, hit.normal, refraction_ratio);
            }
            result.scattered = Ray(hit.point, direction);
            result.did_scatter = true;
        }
        default: {
            // Emissive - doesn't scatter
            result.did_scatter = false;
        }
    }
    
    return result;
}

fn trace(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var current_ray = ray;
    var color = vec3<f32>(1.0);
    
    for (var depth = 0u; depth < uniforms.max_depth; depth++) {
        let hit = hit_scene(current_ray, EPSILON, 1e10);
        
        if hit.t < 0.0 {
            // Sky background
            let unit_dir = normalize(current_ray.direction);
            let t = 0.5 * (unit_dir.y + 1.0);
            let sky = (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
            return color * sky;
        }
        
        // Emission
        if hit.material.material_type == 3u {
            return color * hit.material.albedo * hit.material.emission;
        }
        
        // Scatter
        let scatter_result = scatter(current_ray, hit, seed);
        if !scatter_result.did_scatter {
            return vec3<f32>(0.0);
        }
        
        color *= scatter_result.attenuation;
        current_ray = scatter_result.scattered;
        
        // Russian roulette
        let p = max(color.x, max(color.y, color.z));
        if depth > 3u {
            if rand(seed) > p {
                return vec3<f32>(0.0);
            }
            color /= p;
        }
    }
    
    return vec3<f32>(0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if x >= uniforms.width || y >= uniforms.height {
        return;
    }
    
    let pixel_idx = y * uniforms.width + x;
    var seed = pixel_idx * 1973u + uniforms.frame * 9277u + 1u;
    
    var color = vec3<f32>(0.0);
    
    for (var s = 0u; s < uniforms.samples; s++) {
        let u = (f32(x) + rand(&seed)) / f32(uniforms.width - 1u);
        let v = (f32(uniforms.height - 1u - y) + rand(&seed)) / f32(uniforms.height - 1u);
        
        let ray = get_camera_ray(u, v, &seed);
        color += trace(ray, &seed);
    }
    
    color /= f32(uniforms.samples);
    
    // Handle accumulation for progressive rendering
    if uniforms.accumulate == 1u {
        let prev = accumulation[pixel_idx].xyz;
        let frame_f = f32(uniforms.frame);
        let accumulated = prev + color;
        accumulation[pixel_idx] = vec4<f32>(accumulated, 1.0);
        let current_frames = frame_f + 1.0;
        color = accumulated / current_frames;
    } else {
        accumulation[pixel_idx] = vec4<f32>(color, 1.0);
    }
    
    // Gamma correction
    color = sqrt(color);
    
    output[pixel_idx] = vec4<f32>(color, 1.0);
}
