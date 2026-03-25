use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use nebula::math::Vec3;
use nebula::geometry::Sphere;
use nebula::scene::Scene;
use nebula::camera::Camera;
use nebula::material::{Lambertian, Metal, Dielectric};
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

criterion_group!(benches, bench_render_small, bench_bvh, bench_aabb);
criterion_main!(benches);
