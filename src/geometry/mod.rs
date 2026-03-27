//! Geometric primitives

mod hit;
mod instance;
mod mesh;
mod sphere;
mod triangle;

pub use hit::{HitRecord, Hittable};
pub use instance::Instance;
pub use mesh::{Mesh, load_obj};
pub use sphere::Sphere;
pub use triangle::Triangle;
