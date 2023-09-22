# stable-fluids-3d
This is a single-file implementation of 3D stable fluids.

![000386](https://github.com/YuCrazing/stable-fluids-3d/assets/8120108/d046e914-782a-41c7-9d77-391daf45b68c)


## Running
Prerequisites:
``` bash
pip3 install taichi
```

Running:
``` bash
python3 stable_fluids_3d.py
```

Default settings:
* `grid resolution`: 128\*128\*128.
* `particle number`: 10 million.
* `jacobi iteration num`: 100.

## Performance

Profiled on NVIDIA RTX 3090:

|#Particle| 1 Million | 5 Million| 
|---      |---        |---       |
|FPS      | 30        | 5        |

## Video
Compressed version:

![video_2_x4_600](https://github.com/YuCrazing/stable-fluids-3d/assets/8120108/5757eb55-3de8-49e3-aa19-16a4d522e9e8)


High-quality version: [YouTube](www.youtube.com).

