"""Source raster units, orientation, provenance and cache integrity."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hooke"))
from worlds.pds_terrain import PRODUCTS, digest, label_fields, load_crop


class PdsTerrain(unittest.TestCase):
    def test_crop_origin_and_dimensions_preserve_declared_geography(self):
        for world, product in PRODUCTS.items():
            height, metadata = load_crop(ROOT / "Hooke/assets/space/terrain" / world)
            fields = metadata["fields"]
            self.assertLessEqual(
                float(fields["MINIMUM_LATITUDE"]), metadata["origin_latitude_deg"]
            )
            self.assertLessEqual(
                metadata["origin_latitude_deg"], float(fields["MAXIMUM_LATITUDE"])
            )
            self.assertLessEqual(
                float(fields["WESTERNMOST_LONGITUDE"]), metadata["origin_longitude_deg"]
            )
            self.assertLessEqual(
                metadata["origin_longitude_deg"], float(fields["EASTERNMOST_LONGITUDE"])
            )
            np.testing.assert_allclose(
                metadata["extent_m"],
                (np.asarray(height.shape) - 1) * float(fields["MAP_SCALE"]),
                rtol=1e-12,
            )
            self.assertEqual(
                metadata["local_frame"], "east_north_up_center_pixel_origin"
            )

    def test_selected_source_ranges_reconstruct_cached_local_pixels_exactly(self):
        for world, product in PRODUCTS.items():
            folder = ROOT / "Hooke/assets/space/terrain" / world
            height, metadata = load_crop(folder)
            header = (folder / "source_label.lbl").read_bytes()
            fields = label_fields(header)
            self.assertEqual(fields["PRODUCT_ID"], product["product_id"])
            self.assertEqual(digest(header), metadata["label_sha256"])
            rows = []
            for row in metadata["downloaded_ranges"]:
                raw = (folder / row["file"]).read_bytes()
                self.assertEqual(digest(raw), row["sha256"])
                self.assertEqual(len(raw), product["count"] * 4)
                rows.append(np.frombuffer(raw, dtype="<f4").astype(float))
            absolute = np.asarray(rows) * metadata["scale"] + metadata["offset"]
            np.testing.assert_array_equal(
                height, np.flipud(absolute - metadata["origin_source_elevation_m"])
            )
            self.assertEqual(height[product["count"] // 2, product["count"] // 2], 0)

    def test_unsupported_units_and_projection_are_rejected(self):
        header = (
            ROOT / "Hooke/assets/space/terrain/martian/source_label.lbl"
        ).read_bytes()
        for altered in (
            header.replace(b"METERS/PIXEL", b"KM/PIXEL"),
            header.replace(b"EQUIRECTANGULAR", b"POLAR_STEREOGRAPHIC"),
            header.replace(b"PC_REAL", b"MSB_INTEGER"),
        ):
            with self.assertRaises(ValueError):
                label_fields(altered)

    def test_modified_height_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.save(folder / "height_m.npy", np.zeros((3, 3)))
            (folder / "manifest.json").write_text(
                json.dumps(dict(height_sha256="incorrect", shape=[3, 3]))
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                load_crop(folder)


if __name__ == "__main__":
    unittest.main()
