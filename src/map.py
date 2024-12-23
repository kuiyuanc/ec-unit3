import os
from enum import IntEnum, auto

import numpy as np
from PIL import Image


class Tile(IntEnum):
    GRASS = auto()
    MOUNTAIN = auto()
    RIVER = auto()
    RIVERSTONE = auto()
    ROCK = auto()


class Map:
    _tile_images = {}
    _tile_size = (64, 64)

    @classmethod
    def load_tile_images(cls, assets_path: str):
        """
        from Claude
        """
        for tile in Tile:
            image_path = os.path.join(assets_path, f"{tile.name.lower()}.png")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                cls._tile_images[tile.value] = image.resize(cls._tile_size)

    def __init__(self, *, width: int, height: int):
        self.shape = (height, width)

        num_tiles = width * height
        self.tiles = np.zeros(num_tiles, dtype=np.uint8)

    def save(self, filename: str = "unnamed", *, path: str = ".", assets_path: str = "assets") -> None:
        # self._save_num(path, filename)
        self._save_image(path, filename, assets_path)

    def _save_num(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)

        fullpath = os.path.join(path, f"{filename}.map")
        grid = self.tiles.reshape(self.shape)
        np.savetxt(fullpath, grid, fmt="%d")

    def _save_image(self, path: str, filename: str, assets_path: str) -> None:
        """
        from Claude
        """
        if not self._tile_images:
            self.load_tile_images(assets_path)

        first_image = next(iter(self._tile_images.values()))
        tile_width, tile_height = first_image.size

        output_width = self.shape[1] * tile_width
        output_height = self.shape[0] * tile_height
        output_image = Image.new("RGBA", (output_width, output_height))

        shaped_map = self.tiles.reshape(self.shape)
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                tile_value = shaped_map[y, x]
                if tile_value in self._tile_images:
                    output_image.paste(self._tile_images[tile_value], (x * tile_width, y * tile_height))

        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, f"{filename}.png")
        output_image.save(output_path)
