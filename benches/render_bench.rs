// Placeholder for benchmarks
use criterion::{criterion_group, criterion_main, Criterion};

fn render_benchmark(_c: &mut Criterion) {
    // TODO: Add benchmarks
}

criterion_group!(benches, render_benchmark);
criterion_main!(benches);
