# stable-fluids-3d
This is a single-file implementation of 3D stable fluids.

![000386](https://github.com/YuCrazing/stable-fluids-3d/assets/8120108/843c3567-b986-49c4-8816-221e083f2c7e)

High-quality video: [YouTube](www.youtube.com).


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
