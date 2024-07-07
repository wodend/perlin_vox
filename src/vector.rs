use glam::{IVec3, UVec2, UVec3, Vec2, Vec3, UVec4};

pub type Size2 = (usize, usize);
pub type Size3 = (usize, usize, usize);
pub type Size4 = (usize, usize, usize, usize);

pub trait Dim3 {
    fn into_vec3(self) -> Vec3;
    fn into_ivec3(self) -> IVec3;
    fn into_uvec3(self) -> UVec3;
}

pub trait Dim2 {
    fn into_vec2(self) -> Vec2;
    fn into_uvec2(self) -> UVec2;
}

pub trait Vector3 {
    fn into_size3(self) -> Size3;
}

pub trait Vector4 {
    fn into_size4(self) -> Size4;
}

impl Dim3 for Size3 {
    fn into_vec3(self) -> Vec3 {
        Vec3::new(self.0 as f32, self.1 as f32, self.2 as f32)
    }

    fn into_ivec3(self) -> IVec3 {
        IVec3::new(self.0 as i32, self.1 as i32, self.2 as i32)
    }

    fn into_uvec3(self) -> UVec3 {
        UVec3::new(self.0 as u32, self.1 as u32, self.2 as u32)
    }
}

impl Dim2 for Size2 {
    fn into_vec2(self) -> Vec2 {
        Vec2::new(self.0 as f32, self.1 as f32)
    }

    fn into_uvec2(self) -> UVec2 {
        UVec2::new(self.0 as u32, self.1 as u32)
    }
}

impl Vector3 for UVec3 {
    fn into_size3(self) -> Size3 {
        (self.x as usize, self.y as usize, self.z as usize)
    }
}

impl Vector4 for UVec4 {
    fn into_size4(self) -> Size4 {
        (self.x as usize, self.y as usize, self.z as usize, self.w as usize)
    }
}
