"""Accelerated AABB culling preserves the original complete candidate set."""
from types import SimpleNamespace
import unittest
import numpy as np
from real_labs.rigid_access import near_geometry_pairs


class BroadphaseTests(unittest.TestCase):
    def test_random_rotated_boxes_match_the_unculled_vector_test(self):
        rng = np.random.default_rng(71)
        for count in (0, 2, 40, 180):
            rotations = []
            for _ in range(count):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                rotations.append(q)
            rotation = np.asarray(rotations).reshape(count, 3, 3)
            data = SimpleNamespace(geom_xmat=rotation.reshape(count, 9), geom_xpos=rng.uniform(-1, 1, (count, 3)))
            model = SimpleNamespace(geom_aabb=np.c_[rng.uniform(-.1, .1, (count, 3)), rng.uniform(.001, .3, (count, 3))])
            pairs = np.array([(i, j) for i in range(count) for j in range(i+1, count)], dtype=int).reshape(-1, 2)
            center = data.geom_xpos + np.einsum('nij,nj->ni', rotation, model.geom_aabb[:, :3])
            half = np.einsum('nij,nj->ni', abs(rotation), model.geom_aabb[:, 3:])
            first, second = pairs.T
            gap = np.linalg.norm(np.maximum(abs(center[first]-center[second])-half[first]-half[second], 0), axis=1)
            for cap in (.000001, .002, .04, 2.):
                np.testing.assert_array_equal(near_geometry_pairs(model, data, pairs, cap), pairs[gap < cap])

    def test_diagonal_and_exact_cap_boundaries_are_preserved(self):
        model = SimpleNamespace(geom_aabb=np.array([[0,0,0,.1,.1,.1]]*4, dtype=float))
        data = SimpleNamespace(geom_xmat=np.tile(np.eye(3).ravel(), (4,1)),
                               geom_xpos=np.array([[0,0,0],[.2015,.2015,0],[.201,0,0],[.202,0,0]]))
        pairs = np.array([[0,1],[0,2],[0,3]])
        center, half = data.geom_xpos, model.geom_aabb[:,3:]
        a,b = pairs.T
        distance = np.linalg.norm(np.maximum(abs(center[a]-center[b])-half[a]-half[b],0),axis=1)
        np.testing.assert_array_equal(near_geometry_pairs(model,data,pairs,.002),pairs[distance<.002])


if __name__ == '__main__':
    unittest.main()
