// Path tracing compute shader

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
    _padding: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> output: array<vec4<f32>>;

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

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.direction * t;
}

fn get_camera_ray(u: f32, v: f32, seed: ptr<function, u32>) -> Ray {
    let origin = uniforms.camera_origin.xyz;
    let direction = uniforms.camera_lower_left.xyz 
        + uniforms.camera_horizontal.xyz * u 
        + uniforms.camera_vertical.xyz * v 
        - origin;
    return Ray(origin, normalize(direction));
}

// Simple sphere intersection
struct HitRecord {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    front_face: bool,
    material_id: u32,
}

fn hit_sphere(center: vec3<f32>, radius: f32, ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
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
    rec.material_id = 0u;
    
    return rec;
}

// Scene: ground sphere + 3 main spheres
fn hit_scene(ray: Ray, t_min: f32, t_max: f32) -> HitRecord {
    var closest: HitRecord;
    closest.t = -1.0;
    var closest_t = t_max;
    
    // Ground
    let ground = hit_sphere(vec3<f32>(0.0, -1000.0, 0.0), 1000.0, ray, t_min, closest_t);
    if ground.t > 0.0 {
        closest = ground;
        closest.material_id = 0u; // gray diffuse
        closest_t = ground.t;
    }
    
    // Center sphere (glass)
    let center = hit_sphere(vec3<f32>(0.0, 1.0, 0.0), 1.0, ray, t_min, closest_t);
    if center.t > 0.0 {
        closest = center;
        closest.material_id = 1u; // glass
        closest_t = center.t;
    }
    
    // Left sphere (diffuse)
    let left = hit_sphere(vec3<f32>(-4.0, 1.0, 0.0), 1.0, ray, t_min, closest_t);
    if left.t > 0.0 {
        closest = left;
        closest.material_id = 2u; // brown diffuse
        closest_t = left.t;
    }
    
    // Right sphere (metal)
    let right = hit_sphere(vec3<f32>(4.0, 1.0, 0.0), 1.0, ray, t_min, closest_t);
    if right.t > 0.0 {
        closest = right;
        closest.material_id = 3u; // metal
        closest_t = right.t;
    }
    
    return closest;
}

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
        
        // Simple diffuse for all materials (simplified)
        let scatter_dir = hit.normal + rand_unit_vector(seed);
        current_ray = Ray(hit.point, normalize(scatter_dir));
        
        // Material colors
        var albedo = vec3<f32>(0.5); // default gray
        if hit.material_id == 1u {
            albedo = vec3<f32>(1.0); // glass (simplified to white)
        } else if hit.material_id == 2u {
            albedo = vec3<f32>(0.4, 0.2, 0.1); // brown
        } else if hit.material_id == 3u {
            albedo = vec3<f32>(0.7, 0.6, 0.5); // metal gold
        }
        
        color *= albedo;
        
        // Russian roulette
        let p = max(color.x, max(color.y, color.z));
        if rand(seed) > p {
            return vec3<f32>(0.0);
        }
        color /= p;
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
    var seed = pixel_idx * 1973u + 9277u;
    
    var color = vec3<f32>(0.0);
    
    for (var s = 0u; s < uniforms.samples; s++) {
        let u = (f32(x) + rand(&seed)) / f32(uniforms.width - 1u);
        let v = (f32(uniforms.height - 1u - y) + rand(&seed)) / f32(uniforms.height - 1u);
        
        let ray = get_camera_ray(u, v, &seed);
        color += trace(ray, &seed);
    }
    
    color /= f32(uniforms.samples);
    
    // Gamma correction
    color = sqrt(color);
    
    output[pixel_idx] = vec4<f32>(color, 1.0);
}
