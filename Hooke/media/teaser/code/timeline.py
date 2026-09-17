# name, start (s), duration (s), voiceover offset within scene (s)
# Part 1 (history) is compressed to 10 s total; part 2 is faster and louder.
SCENES = [
    ("s1", 0.0, 4.1, 0.05),
    ("s2", 4.1, 2.0, 0.05),
    ("s3", 6.1, 1.5, None),
    ("s4", 7.6, 2.4, 0.05),
    ("s5", 10.0, 7.0, 0.9),
    ("s6", 17.0, 5.6, 0.15),
    ("s7", 22.6, 5.0, 0.1),
    ("s8", 27.6, 7.0, 1.0),
    ("end", 34.6, 4.4, None),
]
TOTAL = 39.0
PART2_START = 10.0
