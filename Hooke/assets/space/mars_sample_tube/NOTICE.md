# Mars 2020 采样管

NASA/JPL-Caltech; cap mesh/material modifications and USD packaging by Space Robotics Bench contributors. Converted for Hooke research simulation.

Source: https://github.com/AndrejOrsula/srb_assets/tree/54b1282a5f89ead8ff51b869060e68083fa3ad2f

Terms: https://www.nasa.gov/nasa-brand-center/images-and-media/

Original NASA 3D resource uses NASA media guidelines. SRB modifications are CC0; the underlying NASA resource is not relabeled CC0. Retain both origins. No JPL/NASA endorsement is implied.

Changes: geometry transformed to a centered XY origin with bottom at Z=0; source triangle/material groups, UV coordinates and normals exported to OBJ; diffuse images converted to lossless RGB PNG; USD linear base colors converted to sRGB for MuJoCo. Full PBR/metallic response is not reproduced. Source physical properties are not inferred from appearance. The XML uses an explicit illustrative 0.1 kg mass and bounding-box inertia, not measured properties; cabin and handrail remain fixed in the scenes.
