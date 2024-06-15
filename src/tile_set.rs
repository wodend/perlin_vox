use ndarray::Array3;

pub struct TileSet {
    pub tile_size: usize,
    tiles: Vec<Array3<[u8; 4]>>,
}

impl TileSet {
    // Create a new TileSet
    pub fn new(tile_size: usize) -> Self {
        Self {
            tile_size: tile_size,
            tiles: Vec::new(),
        }
    }

    // Insert a new tile and return its ID
    pub fn insert(&mut self, varray3: Array3<[u8; 4]>) -> std::io::Result<usize> {
        self.check_size(varray3.dim())?;
        self.tiles.push(varray3);
        Ok(self.tiles.len() - 1)
    }

    fn check_size(&self, tile_size: (usize, usize, usize)) -> std::io::Result<()> {
        if tile_size.0 != self.tile_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "tile x size is not equal to tile set tile size",
            ));
        }
        if tile_size.1 != self.tile_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "tile y size is not equal to tile set tile size",
            ));
        }
        if tile_size.2 != self.tile_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "tile z size is not equal to tile set tile size",
            ));
        }
        Ok(())
    }

    // Get the varray values for a given ID
    pub fn get(&self, id: usize) -> Option<&Array3<[u8; 4]>> {
        self.tiles.get(id)
    }

    /// Get the number of tiles in the TileSet
    pub fn len(&self) -> usize {
        self.tiles.len()
    }
}