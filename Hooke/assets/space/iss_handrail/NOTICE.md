# ISS 扶手

NASA Astrobee project; ISS handrails from NASA Robonaut2. Converted for Hooke research simulation.

Source: https://github.com/nasa/astrobee_media/tree/1cb0620121099cc848f5a3db480464188dc91ea3

Terms: https://www.nasa.gov/nasa-brand-center/images-and-media/

Retain source credit and third-party notices. NASA media terms cover this asset per the upstream README; do not relabel it CC0 or imply NASA endorsement. Logos and separate third-party rights are not granted by these terms.

Changes: geometry transformed to a centered XY origin with bottom at Z=0; source triangle/material groups, UV coordinates and normals exported to OBJ; diffuse images converted to lossless RGB PNG; USD linear base colors converted to sRGB for MuJoCo. Full PBR/metallic response is not reproduced. Source physical properties are not inferred from appearance. The XML uses an explicit illustrative 0.1 kg mass and bounding-box inertia, not measured properties; cabin and handrail remain fixed in the scenes.
