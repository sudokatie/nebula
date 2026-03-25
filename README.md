# nebula

A physically-based path tracer with CPU and GPU rendering.

## Features

- Unbiased Monte Carlo path tracing
- Materials: Lambertian (diffuse), Metal, Dielectric (glass), Emissive
- BVH acceleration with optimized AABB intersection
- Depth of field camera
- JSON scene format
- CPU rendering with rayon parallelism
- GPU rendering with wgpu compute shaders (optional)

## Quick Start

```bash
# Build
cargo build --release

# Render a scene
./target/release/nebula render scenes/spheres.json -o output.png

# With options
./target/release/nebula render scene.json \
  --width 1920 --height 1080 \
  --samples 100 --depth 50

# GPU rendering (requires --features gpu)
cargo build --release --features gpu
./target/release/nebula render scene.json --gpu
```

## Scene Format

Scenes are JSON files with camera, materials, and objects:

```json
{
  "camera": {
    "position": [0, 1, 3],
    "look_at": [0, 0, 0],
    "fov": 60,
    "aperture": 0.1,
    "focus_distance": 3
  },
  "materials": [
    {"name": "ground", "type": "lambertian", "albedo": [0.5, 0.5, 0.5]},
    {"name": "glass", "type": "dielectric", "ior": 1.5},
    {"name": "gold", "type": "metal", "albedo": [0.8, 0.6, 0.2], "roughness": 0.1}
  ],
  "objects": [
    {"type": "sphere", "center": [0, 1, 0], "radius": 1, "material": "glass"}
  ]
}
```

## Material Types

- **lambertian**: Diffuse material with `albedo` color
- **metal**: Reflective with `albedo` and `roughness` (0-1)
- **dielectric**: Glass/water with `ior` (index of refraction, typically 1.5)
- **emissive**: Light source with `color` and `strength`

## Architecture

```
nebula/
├── src/
│   ├── math/          # Vec3, Ray, Transform
│   ├── geometry/      # Sphere, Triangle, HitRecord
│   ├── accel/         # AABB, BVH
│   ├── material/      # Lambertian, Metal, Dielectric, Emissive
│   ├── camera/        # Thin lens camera with DOF
│   ├── integrator/    # Path tracing
│   ├── scene/         # Scene container and JSON loader
│   ├── render/        # CPU and GPU renderers
│   └── output/        # PPM, PNG, tone mapping
├── shaders/           # WGSL compute shaders
└── scenes/            # Example scenes
```

## Performance

- Branch-free AABB intersection
- Precomputed inverse ray direction
- BVH with median split
- Rayon for CPU parallelism
- wgpu compute shaders for GPU

## License

MIT

---

*Built by Katie, who finds rendering photons weirdly satisfying.*
