use std::collections::HashMap;
use std::fs::{write, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use glam::{IVec3, UVec3};
use ndarray::Array3;

use crate::vector::{Dim3, Vector3};

pub struct Vox {
    size: IVec3,
    xyzis: Vec<[u8; 4]>,
    palette: HashMap<[u8; 4], u8>,
}

impl Vox {
    // Vox spec: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
    const ZERO: [u8; 4] = [0; 4];
    const TAG_SIZE: usize = 4;
    const VOX_TAG: &'static [u8; Self::TAG_SIZE] = b"VOX ";
    const INT_SIZE: usize = 4;
    const VERSION: i32 = 150;

    const MAIN_TAG: &'static [u8; Self::TAG_SIZE] = b"MAIN";
    const CHUNK_BYTE_COUNTS_SIZE: usize = 8;

    const SIZE_TAG: &'static [u8; Self::TAG_SIZE] = b"SIZE";

    const XYZI_TAG: &'static [u8; Self::TAG_SIZE] = b"XYZI";
    const XYZI_SIZE: usize = 4;

    const RGBA_TAG: &'static [u8; Self::TAG_SIZE] = b"RGBA";
    const PALETTE_RGBA_COUNT: usize = 256;
    const RGBA_SIZE: usize = 4;

    pub fn new() -> Vox {
        Vox {
            size: IVec3::new(0, 0, 0),
            xyzis: Vec::new(),
            palette: HashMap::new(),
        }
    }

    pub fn from(varray: &Array3<[u8; 4]>) -> std::io::Result<Vox> {
        // Calculate voxel position and color index data
        let varray_size = varray.dim();
        Self::check_size(varray_size)?;
        let size = varray_size.into_ivec3();
        let mut color_index = 1;
        let mut xyzis = Vec::new();
        let mut palette = HashMap::new();
        for (pos, rgba) in varray.indexed_iter() {
            // MagicaVoxel does not render transparency, here we exclude voxels with alpha=0
            if rgba[3] > 0 {
                let vox = pos.into_uvec3();
                Self::check_xyz(vox)?;
                let mut xyzi = [0; 4];
                xyzi[0] = vox.x as u8;
                xyzi[1] = vox.y as u8;
                xyzi[2] = vox.z as u8;
                match palette.get(rgba) {
                    None => {
                        palette.insert(*rgba, color_index);
                        xyzi[3] = color_index;
                        color_index += 1;
                    }
                    Some(i) => {
                        xyzi[3] = *i as u8;
                    }
                }
                xyzis.push(xyzi);
            }
        }
        Ok(Vox {
            size: size,
            xyzis: xyzis,
            palette: palette,
        })
    }

    pub fn into_varray(&self) -> Array3<[u8; 4]> {
        let mut render = Array3::from_elem(self.size.as_uvec3().into_size3(), [0; 4]);
        for [x, y, z, i] in &self.xyzis {
            render[[*x as usize, *y as usize, *z as usize]] = *self
                .palette
                .iter()
                .find_map(|(key, &val)| if val == *i { Some(key) } else { None })
                .unwrap();
        }
        render
    }

    fn check_size(size: (usize, usize, usize)) -> std::io::Result<()> {
        if size.0 > i32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "vox x size exceeds i32 max",
            ));
        }
        if size.1 > i32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "vox y size exceeds i32 max",
            ));
        }
        if size.2 > i32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "vox z size exceeds i32 max",
            ));
        }
        if size.0 * size.1 * size.2 > i32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "voxel count exceeds i32 max",
            ));
        }
        Ok(())
    }

    fn check_xyz(pos: UVec3) -> std::io::Result<()> {
        if pos.x > u8::MAX as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "voxel x position exceeds u8 max",
            ));
        }
        if pos.y > u8::MAX as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "voxel y position exceeds u8 max",
            ));
        }
        if pos.z > u8::MAX as u32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "voxel z position exceeds u8 max",
            ));
        }
        Ok(())
    }

    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let vox_tag = Self::read_tag(&mut reader)?;
        Self::check_tag(*Self::VOX_TAG, vox_tag)?;
        let version = Self::read_int(&mut reader)?;
        Self::check_version(version)?;

        let main_tag = Self::read_tag(&mut reader)?;
        Self::check_tag(*Self::MAIN_TAG, main_tag)?;
        reader.consume(Self::CHUNK_BYTE_COUNTS_SIZE);

        let size_tag = Self::read_tag(&mut reader)?;
        Self::check_tag(*Self::SIZE_TAG, size_tag)?;
        reader.consume(Self::CHUNK_BYTE_COUNTS_SIZE);
        let x_size = Self::read_int(&mut reader)?;
        let y_size = Self::read_int(&mut reader)?;
        let z_size = Self::read_int(&mut reader)?;

        let xyzi_tag = Self::read_tag(&mut reader)?;
        Self::check_tag(*Self::XYZI_TAG, xyzi_tag)?;
        reader.consume(Self::CHUNK_BYTE_COUNTS_SIZE);
        let voxel_count = Self::read_int(&mut reader)?;
        let mut xyzis = vec![[0; 4]; voxel_count as usize];
        for xyzi in xyzis.iter_mut() {
            *xyzi = Self::read_xyzi(&mut reader)?;
        }

        let rgba_tag = Self::read_tag(&mut reader)?;
        Self::check_tag(*Self::RGBA_TAG, rgba_tag)?;
        reader.consume(Self::CHUNK_BYTE_COUNTS_SIZE);
        let mut palette = HashMap::new();
        for i in 1..Self::PALETTE_RGBA_COUNT {
            let rgba = Self::read_rgba(&mut reader)?;
            palette.insert(rgba, i as u8);
        }
        let vox = Self {
            size: IVec3::new(x_size, y_size, z_size),
            xyzis: xyzis,
            palette: palette,
        };
        return Ok(vox);
    }

    fn read_tag(reader: &mut BufReader<File>) -> std::io::Result<[u8; Self::TAG_SIZE]> {
        let mut tag = [0; Self::TAG_SIZE];
        reader.read(&mut tag)?;
        Ok(tag)
    }

    fn read_int(reader: &mut BufReader<File>) -> std::io::Result<i32> {
        let mut int_bytes = [0; Self::INT_SIZE];
        reader.read(&mut int_bytes)?;
        Ok(i32::from_le_bytes(int_bytes))
    }

    fn read_xyzi(reader: &mut BufReader<File>) -> std::io::Result<[u8; Self::XYZI_SIZE]> {
        let mut xyzi = [0; Self::XYZI_SIZE];
        reader.read(&mut xyzi)?;
        Ok(xyzi)
    }

    fn read_rgba(reader: &mut BufReader<File>) -> std::io::Result<[u8; Self::RGBA_SIZE]> {
        let mut rgba = [0; Self::RGBA_SIZE];
        reader.read(&mut rgba)?;
        Ok(rgba)
    }

    fn check_tag(expected: [u8; Self::TAG_SIZE], got: [u8; Self::TAG_SIZE]) -> std::io::Result<()> {
        if expected != got {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("expected tag {:?} got {:?}", expected, got),
            ));
        } else {
            Ok(())
        }
    }

    fn check_version(got: i32) -> std::io::Result<()> {
        if Self::VERSION != got {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("expected version {:?} got {:?}", Self::VERSION, got),
            ));
        } else {
            Ok(())
        }
    }

    pub fn write<P>(&self, path: P) -> std::io::Result<()>
    where
        P: AsRef<Path>,
    {
        let bytes = self.magicavoxel_bytes()?;
        write(path, &bytes)?;
        Ok(())
    }

    fn magicavoxel_bytes(&self) -> std::io::Result<Vec<u8>> {
        // Calculate vox chunk size
        let voxel_count = self.xyzis.len() as i32;
        let xyzi_chunk_size = Self::INT_SIZE as i32 + (voxel_count * Self::INT_SIZE as i32);

        // Calculate rgba chunk size
        let rgba_chunk_size = Self::PALETTE_RGBA_COUNT * Self::INT_SIZE;

        // Calculate header data
        let size_chunk_size = Self::INT_SIZE as i32 * 3;
        let chunk_header_size = Self::INT_SIZE as i32 * 3;
        let chunk_count = 3;
        let main_child_chunks_size = (chunk_header_size * chunk_count)
            + size_chunk_size
            + xyzi_chunk_size
            + rgba_chunk_size as i32;

        // Write vox data to bytes
        let mut bytes = Vec::new();
        bytes.write(b"VOX ")?;
        bytes.write(&i32::to_le_bytes(150))?;
        bytes.write(b"MAIN")?;
        bytes.write(&Self::ZERO)?; // MAIN has no content
        bytes.write(&i32::to_le_bytes(main_child_chunks_size))?;
        bytes.write(b"SIZE")?;
        bytes.write(&i32::to_le_bytes(size_chunk_size))?;
        bytes.write(&Self::ZERO)?; // SIZE has no children
        bytes.write(&i32::to_le_bytes(self.size.x))?;
        bytes.write(&i32::to_le_bytes(self.size.y))?;
        bytes.write(&i32::to_le_bytes(self.size.z))?;
        bytes.write(b"XYZI")?;
        bytes.write(&i32::to_le_bytes(xyzi_chunk_size))?;
        bytes.write(&Self::ZERO)?; // XYZI has no children
        bytes.write(&i32::to_le_bytes(voxel_count))?;
        for xyzi in &self.xyzis {
            bytes.write(xyzi)?;
        }
        bytes.write(b"RGBA")?;
        bytes.write(&i32::to_le_bytes(rgba_chunk_size as i32))?;
        bytes.write(&Self::ZERO)?; // RGBA has no children
        let mut palette_buf = [[0; 4]; Self::PALETTE_RGBA_COUNT];
        for (rgba, i) in &self.palette {
            palette_buf[*i as usize - 1] = *rgba;
        }
        bytes.write(&palette_buf.concat())?;
        Ok(bytes)
    }
}
