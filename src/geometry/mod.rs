//! Geometric primitives

mod hit;
mod sphere;
mod triangle;
mod quad;
mod mesh;

pub use hit::{HitRecord, Hittable};
pub use sphere::Sphere;
pub use triangle::Triangle;
pub use quad::Quad;
pub use mesh::Mesh;
