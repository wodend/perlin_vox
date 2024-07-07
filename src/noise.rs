use glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles};
use ndarray::{ArrayBase, Array1, Array2, Array3, Dimension, OwnedRepr};

use crate::vector::{Dim2, Dim3};

/// Normalize an ndarray of f32.
pub fn normalize<D>(narray: &mut ArrayBase<OwnedRepr<f32>, D>)
where
    D: Dimension,
{
    let min = *narray
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max = *narray
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    narray.mapv_inplace(|x| normalize_scalar(x, min, max));
}

/// Normalize an f32 within the given range.
fn normalize_scalar(x: f32, min: f32, max: f32) -> f32 {
    (x - min) / (max - min)
}

/// 1D Perlin noise and its analytical derivative.
pub struct Perlin1 {
    pub values: Array1<f32>,
    pub derivatives: Array1<f32>,
}
impl Perlin1 {
    /// Generate a 1D array of Perlin noise.
    pub fn gen(size: usize, seed: u32, scale: f32) -> Perlin1 {
        let mut values = Array1::from_elem(size, 0.0);
        let mut derivatives = Array1::from_elem(size, 0.0);

        for ((x, n), d) in values.indexed_iter_mut().zip(derivatives.iter_mut()) {
            let p = x as f32 * scale;
            let noise = Self::noise_d(seed, p);
            *n = noise.0;
            *d = noise.1;
        }

        Perlin1 {
            values,
            derivatives,
        }
    }

    /// Generate 1D Perlin noise and its analytical derivative.
    fn noise_d(seed: u32, p: f32) -> (f32, f32) {
        // Unit line lower bound.
        let i = p.floor();
        // Position along unit line.
        let f = p.fract();

        // Gradient scalars for unit line endpoints.
        let ga = Self::hash(seed, i + 0.0);
        let gb = Self::hash(seed, i + 1.0);

        // Projections from unit line endpoints.
        let va = ga * (f - 0.0);
        let vb = gb * (f - 1.0);

        // Fade curve position.
        let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        let u_d = 30.0 * f * f * (f * (f - 2.0) + 1.0);

        // Interpolated value and derivative.
        let v = va + u * (vb - va);
        let v_d = ga + u * (gb - ga) + u_d * (vb - va);

        (v, v_d)
    }

    /// Generate a psuedo random 1D scalar hash.
    fn hash(seed: u32, p: f32) -> f32 {
        let s = seed as f32;
        let p = p * (s + 127.1);

        (p.sin() * 43758.5453).fract()
    }

    pub fn normalize(&mut self) {
        normalize(&mut self.values);
    }
}

/// 2D Perlin noise and its analytical derivative.
pub struct Perlin2 {
    pub values: Array2<f32>,
    pub derivatives: Array2<Vec2>,
}
impl Perlin2 {
    /// Generate a 2D array of Perlin noise.
    pub fn gen(size: (usize, usize), seed: u32, scale: f32) -> Perlin2 {
        let mut values = Array2::from_elem(size, 0.0);
        let mut derivatives = Array2::from_elem(size, Vec2::ZERO);

        for ((xy, v), d) in values.indexed_iter_mut().zip(derivatives.iter_mut()) {
            let p = xy.into_vec2() * scale;
            let noise = Self::noise_d(seed, p);
            *v = noise.0;
            *d = noise.1;
        }

        Perlin2 {
            values,
            derivatives,
        }
    }

    /// Generate 2D Perlin noise and its analytical derivative.
    fn noise_d(seed: u32, p: Vec2) -> (f32, Vec2) {
        // Unit square lower bound.
        let i = p.floor();
        // Position within unit square.
        let f = p.fract();

        // Gradient vectors for unit square verticies.
        let ga = Self::hash(seed, i + Vec2::new(0.0, 0.0));
        let gb = Self::hash(seed, i + Vec2::new(1.0, 0.0));
        let gc = Self::hash(seed, i + Vec2::new(0.0, 1.0));
        let gd = Self::hash(seed, i + Vec2::new(1.0, 1.0));

        // Projections from unit square verticies.
        let va = ga.dot(f - Vec2::new(0.0, 0.0));
        let vb = gb.dot(f - Vec2::new(1.0, 0.0));
        let vc = gc.dot(f - Vec2::new(0.0, 1.0));
        let vd = gd.dot(f - Vec2::new(1.0, 1.0));

        // Fade curve position.
        let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        let u_d = 30.0 * f * f * (f * (f - 2.0) + 1.0);

        // Interpolated value and derivative.
        let v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
        let v_d = ga
            + u.x * (gb - ga)
            + u.y * (gc - ga)
            + u.x * u.y * (ga - gb - gc + gd)
            + u_d * (u.yx() * (va - vb - vc + vd) + Vec2::new(vb, vc) - va);

        (v, v_d)
    }

    /// Generate a psuedo random 2D vector hash.
    fn hash(seed: u32, p: Vec2) -> Vec2 {
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

    pub fn normalize(&mut self) {
        normalize(&mut self.values);
    }
}

/// 3D Perlin noise and its analytical derivative.
pub struct Perlin3 {
    pub values: Array3<f32>,
    pub derivatives: Array3<Vec3>,
}
impl Perlin3 {
    /// Generate a 3D array of Perlin noise.
    pub fn gen(size: (usize, usize, usize), seed: u32, scale: f32) -> Perlin3 {
        let mut values = Array3::from_elem(size, 0.0);
        let mut derivatives = Array3::from_elem(size, Vec3::ZERO);

        for ((xyz, v), d) in values.indexed_iter_mut().zip(derivatives.iter_mut()) {
            let p = xyz.into_vec3() * scale;
            let noise = Self::noise_d(seed, p);
            *v = noise.0;
            *d = noise.1;
        }

        Perlin3 {
            values,
            derivatives,
        }
    }

    /// Generate 3D Perlin noise and its analytical derivative.
    fn noise_d(seed: u32, p: Vec3) -> (f32, Vec3) {
        // Unit cube lower bound.
        let i = p.floor();
        // Position within unit cube.
        let f = p.fract();

        // Gradient vectors for unit cube verticies.
        let ga = Self::hash(seed, i + Vec3::new(0.0, 0.0, 0.0));
        let gb = Self::hash(seed, i + Vec3::new(1.0, 0.0, 0.0));
        let gc = Self::hash(seed, i + Vec3::new(0.0, 1.0, 0.0));
        let gd = Self::hash(seed, i + Vec3::new(1.0, 1.0, 0.0));
        let ge = Self::hash(seed, i + Vec3::new(0.0, 0.0, 1.0));
        let gf = Self::hash(seed, i + Vec3::new(1.0, 0.0, 1.0));
        let gg = Self::hash(seed, i + Vec3::new(0.0, 1.0, 1.0));
        let gh = Self::hash(seed, i + Vec3::new(1.0, 1.0, 1.0));

        // Projections from unit cube verticies.
        let va = ga.dot(f - Vec3::new(0.0, 0.0, 0.0));
        let vb = gb.dot(f - Vec3::new(1.0, 0.0, 0.0));
        let vc = gc.dot(f - Vec3::new(0.0, 1.0, 0.0));
        let vd = gd.dot(f - Vec3::new(1.0, 1.0, 0.0));
        let ve = ge.dot(f - Vec3::new(0.0, 0.0, 1.0));
        let vf = gf.dot(f - Vec3::new(1.0, 0.0, 1.0));
        let vg = gg.dot(f - Vec3::new(0.0, 1.0, 1.0));
        let vh = gh.dot(f - Vec3::new(1.0, 1.0, 1.0));

        // Fade curve position.
        let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        let u_d = 30.0 * f * f * (f * (f - 2.0) + 1.0);

        // Interpolated value and derivative.
        let v = va
            + u.x * (vb - va)
            + u.y * (vc - va)
            + u.z * (ve - va)
            + u.x * u.y * (va - vb - vc + vd)
            + u.y * u.z * (va - vc - ve + vg)
            + u.z * u.x * (va - vb - ve + vf)
            + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
        let v_d = ga
            + u.x * (gb - ga)
            + u.y * (gc - ga)
            + u.z * (ge - ga)
            + u.x * u.y * (ga - gb - gc + gd)
            + u.y * u.z * (ga - gc - ge + gg)
            + u.z * u.x * (ga - gb - ge + gf)
            + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
            + u_d
                * (Vec3::new(vb - va, vc - va, ve - va)
                    + u.yzx() * Vec3::new(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
                    + u.zxy() * Vec3::new(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
                    + u.yzx() * u.zxy() * (-va + vb + vc - vd + ve - vf - vg + vh));

        (v, v_d)
    }

    /// Generate a psuedo random 3D vector hash.
    fn hash(seed: u32, p: Vec3) -> Vec3 {
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

    pub fn normalize(&mut self) {
        normalize(&mut self.values);
    }
}
