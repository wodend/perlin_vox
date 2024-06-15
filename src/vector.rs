use glam::{UVec2, Vec2, IVec3, UVec3, Vec3};

pub trait Pos3 {
    fn into_vec3(self) -> Vec3;
    fn into_ivec3(self) -> IVec3;
    fn into_uvec3(self) -> UVec3;
}

pub trait Pos2 {
    fn into_vec2(self) -> Vec2;
    fn into_uvec2(self) -> UVec2;
}

pub trait Vector3 {
    fn into_pos(self) -> (usize, usize, usize);
}

impl Pos3 for (usize, usize, usize) {
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

impl Pos2 for (usize, usize) {
    fn into_vec2(self) -> Vec2 {
        Vec2::new(self.0 as f32, self.1 as f32)
    }

    fn into_uvec2(self) -> UVec2 {
        UVec2::new(self.0 as u32, self.1 as u32)
    }
}

impl Vector3 for UVec3 {
    fn into_pos(self) -> (usize, usize, usize) {
        (self.x as usize, self.y as usize, self.z as usize)
    }
}
