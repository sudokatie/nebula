// Path tracing compute shader with scene data support

// GPU limits
const MAX_SPHERES: u32 = 1024u;
const MAX_TRIANGLES: u32 = 65536u;
const MAX_MATERIALS: u32 = 256u;

// Material types
const MAT_LAMBERTIAN: u32 = 0u;
const MAT_METAL: u32 = 1u;
const MAT_DIELECTRIC: u32 = 2u;
const MAT_EMISSIVE: u32 = 3u;

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
    sample_offset: u32,
    sphere_count: u32,
    triangle_count: u32,
}

struct Sphere {
    center: vec3<f32>,
    radius: f32,
    material_id: u32,
    _padding: vec3<u32>,
}

struct Triangle {
    v0: vec3<f32>,
    _pad0: f32,
    v1: vec3<f32>,
    _pad1: f32,
    v2: vec3<f32>,
    material_id: u32,
}

struct Material {
    albedo: vec3<f32>,
    material_type: u32,
    roughness: f32,
    ior: f32,
    emission_strength: f32,
    _padding: f32,
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
var<storage, read> materials: array<Material>;

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

struct HitRecord {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    material_id: u32,
}

// Sphere intersection
fn hit_sphere(sphere_idx: u32, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    let sphere = spheres[sphere_idx];
    let oc = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let half_b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
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
    let outward_normal = (rec.point - sphere.center) / sphere.radius;
    rec.front_face = dot(ray.direction, outward_normal) < 0.0;
    rec.normal = select(-outward_normal, outward_normal, rec.front_face);
    rec.material_id = sphere.material_id;
    
    return rec;
}

// Triangle intersection (Moller-Trumbore)
fn hit_triangle(tri_idx: u32, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    let tri = triangles[tri_idx];
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    
    var rec: HitRecord;
    rec.t = -1.0;
    
    if abs(a) < 1e-8 {
        return rec;
    }
    
    let f = 1.0 / a;
    let s = ray.origin - tri.v0;
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
    
    if t < t_min || t > t_max {
        return rec;
    }
    
    rec.t = t;
    rec.point = ray_at(ray, t);
    let outward_normal = normalize(cross(edge1, edge2));
    rec.front_face = dot(ray.direction, outward_normal) < 0.0;
    rec.normal = select(-outward_normal, outward_normal, rec.front_face);
    rec.material_id = tri.material_id;
    
    return rec;
}

// Scene intersection
fn hit_scene(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest: HitRecord;
    closest.t = -1.0;
    var closest_t = t_max;
    
    // Test spheres
    for (var i = 0u; i < uniforms.sphere_count && i < MAX_SPHERES; i++) {
        let rec = hit_sphere(i, ray, t_min, closest_t);
        if rec.t > 0.0 {
            closest = rec;
            closest_t = rec.t;
        }
    }
    
    // Test triangles
    for (var i = 0u; i < uniforms.triangle_count && i < MAX_TRIANGLES; i++) {
        let rec = hit_triangle(i, ray, t_min, closest_t);
        if rec.t > 0.0 {
            closest = rec;
            closest_t = rec.t;
        }
    }
    
    return closest;
}

// Schlick approximation for Fresnel
fn schlick(cosine: f32, ior: f32) -> f32 {
    var r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// Reflect vector
fn reflect_vec(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

// Refract vector
fn refract_vec(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_out_perp = eta * (v + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

// Trace ray through scene
fn trace(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var current_ray = ray;
    var color = vec3<f32>(1.0);
    
    for (var depth = 0u; depth < uniforms.max_depth; depth++) {
        let hit = hit_scene(current_ray, 0.001, 1e10);
        
        if hit.t < 0.0 {
            // Sky background
            let unit_dir = normalize(current_ray.direction);
            let t = 0.5 * (unit_dir.y + 1.0);
            let sky = (1.0 - t) * vec3<f32>(1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
            return color * sky;
        }
        
        let mat = materials[hit.material_id];
        
        // Handle different material types
        switch mat.material_type {
            case MAT_EMISSIVE: {
                return color * mat.albedo * mat.emission_strength;
            }
            case MAT_METAL: {
                let reflected = reflect_vec(normalize(current_ray.direction), hit.normal);
                let scattered = reflected + mat.roughness * rand_in_unit_sphere(seed);
                if dot(scattered, hit.normal) <= 0.0 {
                    return vec3<f32>(0.0);
                }
                current_ray = Ray(hit.point, normalize(scattered));
                color *= mat.albedo;
            }
            case MAT_DIELECTRIC: {
                let refraction_ratio = select(mat.ior, 1.0 / mat.ior, hit.front_face);
                let unit_dir = normalize(current_ray.direction);
                let cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
                let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                
                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let should_reflect = schlick(cos_theta, refraction_ratio) > rand(seed);
                
                var direction: vec3<f32>;
                if cannot_refract || should_reflect {
                    direction = reflect_vec(unit_dir, hit.normal);
                } else {
                    direction = refract_vec(unit_dir, hit.normal, refraction_ratio);
                }
                current_ray = Ray(hit.point, direction);
                // Glass doesn't absorb light
            }
            default: { // MAT_LAMBERTIAN
                let scatter_dir = hit.normal + rand_unit_vector(seed);
                current_ray = Ray(hit.point, normalize(scatter_dir));
                color *= mat.albedo;
            }
        }
        
        // Russian roulette
        let p = max(color.x, max(color.y, color.z));
        if depth > 3u && rand(seed) > p {
            return vec3<f32>(0.0);
        }
        if depth > 3u {
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
    var seed = pixel_idx * 1973u + uniforms.sample_offset * 9277u;
    
    var color = vec3<f32>(0.0);
    
    for (var s = 0u; s < uniforms.samples; s++) {
        let u = (f32(x) + rand(&seed)) / f32(uniforms.width - 1u);
        let v = (f32(uniforms.height - 1u - y) + rand(&seed)) / f32(uniforms.height - 1u);
        
        let ray = get_camera_ray(u, v, &seed);
        color += trace(ray, &seed);
    }
    
    color /= f32(uniforms.samples);
    
    // Accumulate for progressive rendering
    let prev = output[pixel_idx].xyz;
    let weight = f32(uniforms.sample_offset) / f32(uniforms.sample_offset + 1u);
    color = mix(color, prev, weight);
    
    // Gamma correction
    color = sqrt(color);
    
    output[pixel_idx] = vec4<f32>(color, 1.0);
}
