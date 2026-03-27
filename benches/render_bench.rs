use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use nebula::math::Vec3;
use nebula::geometry::{Sphere, Triangle};
use nebula::scene::Scene;
use nebula::camera::Camera;
use nebula::material::{Lambertian, Metal, Dielectric, Emissive};
use nebula::render::CpuRenderer;

fn create_test_scene() -> (Scene, Camera) {
    let mut scene = Scene::new();
    
    // Add materials
    let ground = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.5, 0.5, 0.5))));
    let glass = scene.add_material(Box::new(Dielectric::new(1.5)));
    let diffuse = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.4, 0.2, 0.1))));
    let metal = scene.add_material(Box::new(Metal::new(Vec3::new(0.7, 0.6, 0.5), 0.0)));
    
    // Add spheres
    scene.add_sphere(Sphere::new(Vec3::new(0.0, -1000.0, 0.0), 1000.0, ground));
    scene.add_sphere(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0, glass));
    scene.add_sphere(Sphere::new(Vec3::new(-4.0, 1.0, 0.0), 1.0, diffuse));
    scene.add_sphere(Sphere::new(Vec3::new(4.0, 1.0, 0.0), 1.0, metal));
    
    scene.build_bvh();
    
    let camera = Camera::new(
        Vec3::new(13.0, 2.0, 3.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
        16.0 / 9.0,
        0.1,
        10.0,
    );
    
    (scene, camera)
}

/// Create Cornell Box scene for performance target testing
/// Target: < 10s CPU at 1024x1024 100spp
fn create_cornell_box() -> (Scene, Camera) {
    let mut scene = Scene::new();
    
    // Materials
    let white = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.73, 0.73, 0.73))));
    let red = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.65, 0.05, 0.05))));
    let green = scene.add_material(Box::new(Lambertian::new(Vec3::new(0.12, 0.45, 0.15))));
    let light = scene.add_material(Box::new(Emissive::new(Vec3::new(1.0, 1.0, 1.0), 15.0)));
    
    // Right wall (green)
    scene.add_triangle(Triangle::new(
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(555.0, 555.0, 0.0),
        Vec3::new(555.0, 555.0, 555.0),
        green,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(555.0, 555.0, 555.0),
        Vec3::new(555.0, 0.0, 555.0),
        green,
    ));
    
    // Left wall (red)
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 555.0),
        Vec3::new(0.0, 555.0, 0.0),
        red,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        Vec3::new(0.0, 555.0, 555.0),
        red,
    ));
    
    // Floor
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 555.0),
        white,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 555.0),
        Vec3::new(0.0, 0.0, 555.0),
        white,
    ));
    
    // Ceiling
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(555.0, 555.0, 555.0),
        Vec3::new(555.0, 555.0, 0.0),
        white,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 555.0, 555.0),
        Vec3::new(555.0, 555.0, 555.0),
        white,
    ));
    
    // Back wall
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 555.0),
        Vec3::new(555.0, 0.0, 555.0),
        Vec3::new(555.0, 555.0, 555.0),
        white,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(0.0, 0.0, 555.0),
        Vec3::new(555.0, 555.0, 555.0),
        Vec3::new(0.0, 555.0, 555.0),
        white,
    ));
    
    // Light
    scene.add_triangle(Triangle::new(
        Vec3::new(213.0, 554.0, 227.0),
        Vec3::new(343.0, 554.0, 227.0),
        Vec3::new(343.0, 554.0, 332.0),
        light,
    ));
    scene.add_triangle(Triangle::new(
        Vec3::new(213.0, 554.0, 227.0),
        Vec3::new(343.0, 554.0, 332.0),
        Vec3::new(213.0, 554.0, 332.0),
        light,
    ));
    
    // Two spheres instead of boxes for simplicity
    scene.add_sphere(Sphere::new(Vec3::new(185.0, 82.5, 168.0), 82.5, white));
    scene.add_sphere(Sphere::new(Vec3::new(368.0, 165.0, 351.0), 165.0, white));
    
    scene.build_bvh();
    
    let camera = Camera::new(
        Vec3::new(278.0, 278.0, -800.0),
        Vec3::new(278.0, 278.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        40.0,
        1.0,
        0.0,
        800.0,
    );
    
    (scene, camera)
}

fn bench_render_small(c: &mut Criterion) {
    let (scene, camera) = create_test_scene();
    
    let mut group = c.benchmark_group("render");
    
    for (width, height, samples) in [(100, 75, 1), (100, 75, 4), (200, 150, 1)] {
        let renderer = CpuRenderer::new(width, height, samples, 10);
        
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}_{}spp", width, height, samples)),
            &(&scene, &camera),
            |b, (scene, camera)| {
                b.iter(|| renderer.render_parallel(scene, camera, 0));
            },
        );
    }
    
    group.finish();
}

/// Benchmark Cornell Box at reduced resolution to estimate full render time
/// Target: 1024x1024 @ 100spp should take < 10s
fn bench_cornell_box(c: &mut Criterion) {
    let (scene, camera) = create_cornell_box();
    
    let mut group = c.benchmark_group("cornell_box");
    group.sample_size(10); // Fewer samples for longer benchmarks
    
    // Benchmark at 256x256 with 10spp (1/64th of target pixels, 1/10th samples)
    // If this takes X seconds, full render should take ~64*10*X = 640X seconds
    // Target: 640X < 10s, so X < 0.015625s (15.6ms)
    let renderer = CpuRenderer::new(256, 256, 10, 50);
    
    group.bench_function("256x256_10spp", |b| {
        b.iter(|| renderer.render_parallel(&scene, &camera, 0));
    });
    
    // More accurate extrapolation benchmark: 512x512 @ 25spp
    // 1/4 pixels, 1/4 samples = 1/16 total work
    // Target: 16X < 10s, so X < 0.625s
    let renderer_512 = CpuRenderer::new(512, 512, 25, 50);
    
    group.bench_function("512x512_25spp", |b| {
        b.iter(|| renderer_512.render_parallel(&scene, &camera, 0));
    });
    
    group.finish();
}

fn bench_bvh(c: &mut Criterion) {
    use nebula::math::Ray;
    use nebula::accel::BVH;
    use nebula::geometry::Hittable;
    
    // Create many spheres for BVH testing
    let spheres: Vec<Box<dyn Hittable>> = (0..100)
        .map(|i| {
            let x = (i % 10) as f32 * 2.0 - 9.0;
            let z = (i / 10) as f32 * 2.0 - 9.0;
            Box::new(Sphere::new(Vec3::new(x, 0.5, z), 0.5, 0)) as Box<dyn Hittable>
        })
        .collect();
    
    let bvh = BVH::new(spheres);
    
    c.bench_function("bvh_100_spheres", |b| {
        let ray = Ray::new(Vec3::new(0.0, 1.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
        b.iter(|| bvh.hit(&ray, 0.001, f32::INFINITY));
    });
}

fn bench_bvh_large(c: &mut Criterion) {
    use nebula::math::Ray;
    use nebula::accel::BVH;
    use nebula::geometry::Hittable;
    
    let mut group = c.benchmark_group("bvh_large");
    group.sample_size(20);
    
    // Create 1000 spheres
    let spheres: Vec<Box<dyn Hittable>> = (0..1000)
        .map(|i| {
            let x = (i % 32) as f32 * 2.0 - 31.0;
            let y = ((i / 32) % 32) as f32 * 2.0;
            let z = (i / 1024) as f32 * 2.0 - 15.0;
            Box::new(Sphere::new(Vec3::new(x, y, z), 0.5, 0)) as Box<dyn Hittable>
        })
        .collect();
    
    let bvh = BVH::new(spheres);
    
    group.bench_function("1000_spheres", |b| {
        let ray = Ray::new(Vec3::new(0.0, 16.0, 50.0), Vec3::new(0.0, 0.0, -1.0));
        b.iter(|| bvh.hit(&ray, 0.001, f32::INFINITY));
    });
    
    group.finish();
}

fn bench_aabb(c: &mut Criterion) {
    use nebula::math::Ray;
    use nebula::accel::AABB;
    
    let aabb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
    let inv_dir = Vec3::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
    
    let mut group = c.benchmark_group("aabb");
    
    group.bench_function("hit", |b| {
        b.iter(|| aabb.hit(&ray, 0.0, f32::INFINITY));
    });
    
    group.bench_function("hit_precomputed", |b| {
        b.iter(|| aabb.hit_precomputed(&ray.origin, &inv_dir, 0.0, f32::INFINITY));
    });
    
    group.finish();
}

fn bench_simd(c: &mut Criterion) {
    use nebula::math::Vec3x4;
    
    let mut group = c.benchmark_group("simd");
    
    let a = Vec3x4::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
        Vec3::new(10.0, 11.0, 12.0),
    );
    let b = Vec3x4::new(
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(0.4, 0.5, 0.6),
        Vec3::new(0.7, 0.8, 0.9),
        Vec3::new(1.0, 1.1, 1.2),
    );
    
    group.bench_function("vec3x4_dot", |bench| {
        bench.iter(|| a.dot(&b));
    });
    
    group.bench_function("vec3x4_add", |bench| {
        bench.iter(|| a + b);
    });
    
    group.bench_function("vec3x4_length_squared", |bench| {
        bench.iter(|| a.length_squared());
    });
    
    group.finish();
}

criterion_group!(
    benches, 
    bench_render_small, 
    bench_cornell_box,
    bench_bvh, 
    bench_bvh_large,
    bench_aabb,
    bench_simd
);
criterion_main!(benches);
