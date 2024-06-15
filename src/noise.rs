use glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4};
use ndarray::{Array1, Array2, Array3};

use crate::vector::{Pos2, Pos3};

pub struct Noise {
    pub values: Array2<f32>,
    pub derivatives: Array2<Vec2>,
}

/// Generate a psuedo random 1D scalar hash.
fn hash1(seed: u32, p: f32) -> f32 {
    let s = seed as f32;
    let p = p * (s + 127.1);

    (p.sin() * 43758.5453).fract()
}

/// Generate a psuedo random 2D vector hash.
fn hash2(seed: u32, p: Vec2) -> Vec2 {
    let s = seed as f32;
    let p = Vec2::new(
        p.dot(Vec2::new(s + 127.1, s + 311.7)),
        p.dot(Vec2::new(s + 269.5, s + 183.3)),
    );

    Vec2::new(
        (p.x.sin() * 43758.5453).fract(),
        (p.y.sin() * 43758.5453).fract(),
    )
}

/// Generate a psuedo random 3D vector hash.
fn hash3(seed: u32, p: Vec3) -> Vec3 {
    let s = seed as f32;
    let p = Vec3::new(
        p.dot(Vec3::new(s + 127.1, s + 311.7, s + 74.7)),
        p.dot(Vec3::new(s + 269.5, s + 183.3, s + 246.1)),
        p.dot(Vec3::new(s + 113.5, s + 271.9, s + 124.6)),
    );

    Vec3::new(
        (p.x.sin() * 43758.5453).fract(),
        (p.y.sin() * 43758.5453).fract(),
        (p.z.sin() * 43758.5453).fract(),
    )
}

/// Generate 1D Perlin noise and its analytical derivative.
fn noised1(p: f32) -> (f32, f32) {
    let i = p.floor();
    let f = p.fract();

    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let seed = 4;
    let ga = hash1(seed, i + 0.0);
    let gb = hash1(seed, i + 1.0);

    let va = ga * (f - 0.0);
    let vb = gb * (f - 1.0);

    let v = va + u * (vb - va);
    let d = ga + u * (gb - ga) + du * (vb - va);

    (v, d)
}

/// Generate 2D Perlin noise and its analytical derivative.
fn noised2(p: Vec2) -> (f32, Vec2) {
    let i = p.floor();
    let f = p.fract();

    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let seed = 4;
    let ga = hash2(seed, i + Vec2::new(0.0, 0.0));
    let gb = hash2(seed, i + Vec2::new(1.0, 0.0));
    let gc = hash2(seed, i + Vec2::new(0.0, 1.0));
    let gd = hash2(seed, i + Vec2::new(1.0, 1.0));

    let va = ga.dot(f - Vec2::new(0.0, 0.0));
    let vb = gb.dot(f - Vec2::new(1.0, 0.0));
    let vc = gc.dot(f - Vec2::new(0.0, 1.0));
    let vd = gd.dot(f - Vec2::new(1.0, 1.0));

    let v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
    let d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx() * (va - vb - vc + vd) + Vec2::new(vb, vc) - va);

    (v, Vec2::new(d.x, d.y))
}

/// Generate 3D Perlin noise and its analytical derivative.
fn noised3(x: Vec3) -> Vec4 {
    let p = x.floor();
    let w = x.fract();

    // Quintic polynomial interpolation function.
    let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    // Quintic polynomial interpolation derivative function.
    let du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

    // Compute vertex coefficients.
    let seed = 4;
    let ga = hash3(seed, p + Vec3::new(0.0, 0.0, 0.0));
    let gb = hash3(seed, p + Vec3::new(1.0, 0.0, 0.0));
    let gc = hash3(seed, p + Vec3::new(0.0, 1.0, 0.0));
    let gd = hash3(seed, p + Vec3::new(1.0, 1.0, 0.0));
    let ge = hash3(seed, p + Vec3::new(0.0, 0.0, 1.0));
    let gf = hash3(seed, p + Vec3::new(1.0, 0.0, 1.0));
    let gg = hash3(seed, p + Vec3::new(0.0, 1.0, 1.0));
    let gh = hash3(seed, p + Vec3::new(1.0, 1.0, 1.0));

    let va = ga.dot(w - Vec3::new(0.0, 0.0, 0.0));
    let vb = gb.dot(w - Vec3::new(1.0, 0.0, 0.0));
    let vc = gc.dot(w - Vec3::new(0.0, 1.0, 0.0));
    let vd = gd.dot(w - Vec3::new(1.0, 1.0, 0.0));
    let ve = ge.dot(w - Vec3::new(0.0, 0.0, 1.0));
    let vf = gf.dot(w - Vec3::new(1.0, 0.0, 1.0));
    let vg = gg.dot(w - Vec3::new(0.0, 1.0, 1.0));
    let vh = gh.dot(w - Vec3::new(1.0, 1.0, 1.0));

    let v = va
        + u.x * (vb - va)
        + u.y * (vc - va)
        + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd)
        + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf)
        + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);

    let d = ga
        + u.x * (gb - ga)
        + u.y * (gc - ga)
        + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd)
        + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf)
        + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
        + du * (Vec3::new(vb - va, vc - va, ve - va)
            + u.yzx() * Vec3::new(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
            + u.zxy() * Vec3::new(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
            + u.yzx() * u.zxy() * (-va + vb + vc - vd + ve - vf - vg + vh));

    Vec4::new(v, d.x, d.y, d.z)
}

/// Generate a 1D array of Perlin noise.
pub fn narray1(size: usize, scale: f32) -> Array1<f32> {
    let mut narray = Array1::from_elem(size, 0.0);

    for (x, noise) in narray.indexed_iter_mut() {
        *noise = noised1(x as f32 * scale).0;
    }

    narray
}


/// Generate a 2D array of Perlin noise.
pub fn narray2(size: (usize, usize), scale: Vec2) -> Noise {
    let mut values = Array2::from_elem(size, 0.0);
    let mut derivatives = Array2::from_elem(size, Vec2::ZERO);

    for ((xy, v), d) in values.indexed_iter_mut().zip(derivatives.iter_mut()) {
        let position = Pos2::into_vec2(xy);
        let noise = noised2(position * scale);
        *v = noise.0;
        *d = noise.1;
    }

    Noise { values, derivatives }
}

/// Generate a 3D array of Perlin noise.
pub fn narray3() -> Array3<f32> {
    let len = 128;
    let mut narray = Array3::from_elem((len, len, len), 0.0);
    let scalar = Vec3::new(0.1, 0.1, 0.1);

    for (xyz, noise) in narray.indexed_iter_mut() {
        let position = Pos3::into_vec3(xyz);
        *noise = noised3(position * scalar).x;
    }

    narray
}
