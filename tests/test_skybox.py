"""Source cube faces keep their identity after environment-map conversion."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Hooke"))
from backends.skybox import sample_cube, cube_to_latlong, srgb_to_linear


class SkyboxTests(unittest.TestCase):
    def test_six_face_axis_sampling_and_panorama_shape(self):
        faces = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        for i in range(6):
            faces[i] = [i, 10 + i, 20 + i]
        axes = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        np.testing.assert_array_equal(
            sample_cube(faces, axes), [[i, 10 + i, 20 + i] for i in range(6)]
        )
        panorama = cube_to_latlong(faces, 16, 8)
        self.assertEqual(panorama.shape, (8, 16, 3))
        self.assertTrue(np.isin(panorama[:, :, 0], np.arange(6)).all())

    def test_display_color_to_linear_known_values(self):
        np.testing.assert_allclose(
            srgb_to_linear([0.0, 0.5, 1.0]), [0.0, 0.21404114048223255, 1.0], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
