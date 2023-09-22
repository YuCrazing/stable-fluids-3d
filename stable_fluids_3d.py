import taichi as ti
import numpy as np
import time



ti.init(arch=ti.cuda, debug=False)


n = 128
dt = 0.03
dx = 1/n

# 1, 2, 3
RK = 3

enable_BFECC = True
enable_clipping = True

recording_video = False


rho = 1
jacobi_iters = 100

velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))

pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
new_pressures = ti.field(dtype=ti.f32, shape=(n, n, n))

divergences = ti.field(dtype=ti.f32, shape=(n, n, n))

pn_max = 1000000
pn_current = 0
rate = 100000
particles = ti.Vector.field(3, dtype=ti.f32, shape=pn_max)
particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=pn_max)
particle_radius = 0.0001

source_center = ti.Vector([0.5, 0.9, 0.5])
source_radius = 0.001
source_velocity = ti.Vector([0.0, -0.005, 0.0])


# cell center
stagger = ti.Vector([0.5, 0.5, 0.5])


@ti.func
def I(i, j, k):
	return ti.Vector([i, j, k])


@ti.kernel
def init_velocity_field():
	for i in ti.grouped(velocities):
		velocities[i] = ti.Vector([0.0, 0.0, 0.0])


color_tables = [
	ti.Vector([236, 238, 129], ti.f32)/255, # yellow
	ti.Vector([141, 223, 203], ti.f32)/255, # green
	ti.Vector([255, 155, 80], ti.f32)/255, # orange
	ti.Vector([255.0, 0.0, 0.0], ti.f32)/255, # red
]

# color_tables = [
# 	ti.Vector([255, 155, 80], ti.f32)/255, # orange
# 	ti.Vector([255.0, 0.0, 0.0], ti.f32)/255, # red
# 	ti.Vector([255, 155, 80], ti.f32)/255, # orange
# 	ti.Vector([255.0, 0.0, 0.0], ti.f32)/255, # red
# ]

@ti.kernel
def init_particles():
	for i in ti.grouped(particles):
		r = ti.sqrt(ti.random()) * source_radius
		a = np.pi * ti.random() * 2
		b = np.pi * ti.random()
		particles[i] = ti.Vector([r * ti.cos(a) * ti.sin(b), r * ti.sin(a) * ti.sin(b), r * ti.cos(b)]) + source_center
		if particles[i].x < source_center.x and particles[i].z < source_center.z:
			particle_colors[i] = color_tables[0]
		elif particles[i].x > source_center.x and particles[i].z < source_center.z:
			particle_colors[i] = color_tables[1]
		elif particles[i].x > source_center.x and particles[i].z > source_center.z:
			particle_colors[i] = color_tables[2]
		else:
			particle_colors[i] = color_tables[3]

		
@ti.func
def clamp(p):
	# clamp p to [0.5*dx, 1-dx+0.5*dx), i.e. clamp cell index to [0, n-1)
	for d in ti.static(range(p.n)):
		p[d] = min(1 - dx + stagger[d]*dx - 1e-4, max(p[d], stagger[d]*dx))
	return p

@ti.func
def sample_trilinear(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)

	d = grid_f - grid_i
	
	return field[ grid_i ] * (1-d.x)*(1-d.y)*(1-d.z) + field[ grid_i+I(1, 0, 0) ] * d.x*(1-d.y)*(1-d.z) + field[ grid_i+I(0, 1, 0) ] * (1-d.x)*d.y*(1-d.z) + field[ grid_i+I(1, 1, 0) ] * d.x*d.y*(1-d.z) + \
	field[grid_i+I(0, 0, 1)]*(1-d.x)*(1-d.y)*(d.z) + field[ grid_i+I(1, 0, 1) ] * d.x*(1-d.y)*(d.z) + field[ grid_i+I(0, 1, 1) ] * (1-d.x)*d.y*(d.z) + field[ grid_i+I(1, 1, 1) ] * d.x*d.y*(d.z)

@ti.func
def sample_min(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	
	return min( field[ grid_i ], field[ grid_i+I(1, 0, 0) ], field[ grid_i+I(0, 1, 0) ], field[ grid_i+I(1, 1, 0) ], field[ grid_i+I(0, 0, 1) ], field[ grid_i+I(1, 0, 1) ], field[ grid_i+I(0, 1, 1) ], field[ grid_i+I(1, 1, 1) ] )


@ti.func
def sample_max(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	
	return max( field[ grid_i ], field[ grid_i+I(1, 0, 0) ], field[ grid_i+I(0, 1, 0) ], field[ grid_i+I(1, 1, 0) ], field[ grid_i+I(0, 0, 1) ], field[ grid_i+I(1, 0, 1) ], field[ grid_i+I(0, 1, 1) ], field[ grid_i+I(1, 1, 1) ] )


@ti.func
def backtrace(velocities, p, dt):

	if ti.static(RK == 1):
		return p - sample_trilinear(velocities, p) * dt
	elif ti.static(RK == 2):
		p_mid = p - sample_trilinear(velocities, p) * dt * 0.5
		return p - sample_trilinear(velocities, p_mid) * dt
	elif ti.static(RK == 3):
		v_p = sample_trilinear(velocities, p)
		p_mid = p - v_p * dt * 0.5
		v_mid = sample_trilinear(velocities, p_mid)
		p_mid_mid = p - v_mid * dt * 0.75
		v_mid_mid = sample_trilinear(velocities, p_mid_mid)
		return p - (2/9 * v_p + 1/3*v_mid + 4/9*v_mid_mid) * dt



@ti.func
def semi_lagrangian(velocities, field, new_field, dt):
	for i in ti.grouped(field):
		p = (i + stagger) * dx
		new_field[i] = sample_trilinear(field, backtrace(velocities, p, dt))



@ti.func
def BFECC(velocities, field, new_field, new_new_field, dt):
	
	semi_lagrangian(velocities, field, new_field, dt)
	semi_lagrangian(velocities, new_field, new_new_field, -dt)

	for i in ti.grouped(field):
		
		new_field[i] = new_field[i] - 0.5 * (new_new_field[i] - field[i])

		if ti.static(enable_clipping):
			
			source_pos = backtrace(velocities, (i + stagger) * dx, dt )
			mi = sample_min(field, source_pos)
			mx = sample_max(field, source_pos)

			for d in ti.static(range(mi.n)):
				if new_field[i][d] < mi[d] or new_field[i][d] > mx[d]:
					new_field[i] = sample_trilinear(field, source_pos)


@ti.kernel
def advect(velocities:ti.template(), field:ti.template(), new_field:ti.template(), new_new_field:ti.template(), dt:ti.f32):

	if ti.static(enable_BFECC):
		BFECC(velocities, field, new_field, new_new_field, dt)
	else:
		semi_lagrangian(velocities, field, new_field, dt)



@ti.kernel
def solve_divergence(velocities:ti.template(), divergences:ti.template()):
	for i, j, k in velocities:
		c = ti.Vector([i + stagger.x, j + stagger.y, k + stagger.z]) * dx
		l = c - ti.Vector([1, 0, 0]) * dx
		r = c + ti.Vector([1, 0, 0]) * dx
		d = c - ti.Vector([0, 1, 0]) * dx
		u = c + ti.Vector([0, 1, 0]) * dx
		b = c - ti.Vector([0, 0, 1]) * dx
		f = c + ti.Vector([0, 0, 1]) * dx
		v_c = sample_trilinear(velocities, c)
		v_l = sample_trilinear(velocities, l).x
		v_r = sample_trilinear(velocities, r).x
		v_d = sample_trilinear(velocities, d).y
		v_u = sample_trilinear(velocities, u).y
		v_b = sample_trilinear(velocities, b).z
		v_f = sample_trilinear(velocities, f).z


		if i == 0: 
			v_l = -v_c.x
		if i == n-1:
			v_r = -v_c.x
		if j == 0:
			v_d = -v_c.y
		if j == n-1:
			v_u = -v_c.y
		if k == 0:
			v_b = -v_c.z
		if k == n-1:
			v_f = -v_c.z

		divergences[i, j, k] = (v_r - v_l + v_u - v_d + v_f - v_b) / (2*dx)


@ti.kernel
def pressure_jacobi(pressures:ti.template(), new_pressures:ti.template()):

	for i, j, k in pressures:
		c = ti.Vector([i + stagger.x, j + stagger.y, k + stagger.z]) * dx
		l = c - ti.Vector([1, 0, 0]) * dx
		r = c + ti.Vector([1, 0, 0]) * dx
		d = c - ti.Vector([0, 1, 0]) * dx
		u = c + ti.Vector([0, 1, 0]) * dx
		b = c - ti.Vector([0, 0, 1]) * dx
		f = c + ti.Vector([0, 0, 1]) * dx
		p_l = sample_trilinear(pressures, l)
		p_r = sample_trilinear(pressures, r)
		p_d = sample_trilinear(pressures, d)
		p_u = sample_trilinear(pressures, u)
		p_b = sample_trilinear(pressures, b)
		p_f = sample_trilinear(pressures, f)

		new_pressures[i, j, k] = ( p_l + p_r + p_d + p_u + p_b + p_f - divergences[i, j, k] * rho / dt * (dx*dx) ) / 6



@ti.kernel
def projection(velocities:ti.template(), pressures:ti.template()):
	for i, j, k in velocities:
		c = ti.Vector([i + stagger.x, j + stagger.y, k + stagger.z]) * dx
		l = c - ti.Vector([1, 0, 0]) * dx
		r = c + ti.Vector([1, 0, 0]) * dx
		d = c - ti.Vector([0, 1, 0]) * dx
		u = c + ti.Vector([0, 1, 0]) * dx
		b = c - ti.Vector([0, 0, 1]) * dx
		f = c + ti.Vector([0, 0, 1]) * dx
		# p_c = sample_trilinear(pressures, c)
		p_l = sample_trilinear(pressures, l)
		p_r = sample_trilinear(pressures, r)
		p_d = sample_trilinear(pressures, d)
		p_u = sample_trilinear(pressures, u)
		p_b = sample_trilinear(pressures, b)
		p_f = sample_trilinear(pressures, f)

		grad_p = ti.Vector([p_r - p_l, p_u - p_d, p_f - p_b]) / (2*dx)

		# if i == 0:
		# 	grad_p.x = (p_r - p_c) / (dx/2)
		# if i == n-1:
		# 	grad_p.x = (p_c - p_l) / (dx/2)
		# if j == 0:
		# 	grad_p.y = (p_u - p_c) / (dx/2)
		# if j == n-1:
		# 	grad_p.y = (p_c - p_d) / (dx/2)

		velocities[i, j, k] = velocities[i, j, k] - grad_p / rho * dt


@ti.kernel
def apply_force_at_point(velocities:ti.template(), pos:ti.template(), r: ti.template(), force:ti.template()):

	dp = force + (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) * 0.001

	for i, j, k in velocities:
		d2 = (ti.Vector([(i+stagger.x)*dx, (j+stagger.y)*dx, (k+stagger.y)*dx]) - pos).norm_sqr()
		radius = 0.2 * r
		velocities[i, j, k] = velocities[i, j, k] + dp * dt * ti.exp(-d2/radius) * 40


@ti.kernel
def update_particles(pn_current:ti.i32):
	for i in range(pn_current):
		new_v = sample_trilinear(velocities, particles[i])
		particles[i] = particles[i] + new_v * dt 



result_dir = "./result"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)


window = ti.ui.Window("Stable Fluids 3D", (800, 800), vsync=False)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

if recording_video:
	camera.position(0.5, 0.5, 1.1)
else:
	camera.position(0.5, 0.5, 1.5)
camera.lookat(0.5, 0.5, 0.5)
scene.set_camera(camera)


init_velocity_field()
init_particles()

frame = 0
while window.running:

	advect(velocities, velocities, new_velocities, new_new_velocities, dt)
	velocities, new_velocities = new_velocities, velocities
	
	apply_force_at_point(velocities, source_center, source_radius, source_velocity)
	solve_divergence(velocities, divergences)

	for i in range(jacobi_iters):
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	projection(velocities, pressures)
	
	if pn_current < pn_max:
		pn_current += rate
	update_particles(pn_current)

	# rendering
	camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
	# camera.lookat(0.5, 0.5, 0.5)
	scene.set_camera(camera)

	scene.point_light(pos=(0, 5, 2), color=(1, 1, 1))
	scene.ambient_light((0.5, 0.5, 0.5))
	scene.particles(particles, radius=particle_radius, color=(1.0, 0.0, 0.0), per_vertex_color=particle_colors, index_count=pn_current)
	canvas.scene(scene)

	if recording_video:
		if frame > 130:
			video_manager.write_frame(window.get_image_buffer_as_numpy())
	window.show()

	frame += 1
	if recording_video:
		if frame % 10 == 0:
			print(frame)
		if frame > 1000:
			break

if recording_video:
	video_manager.make_video(gif=True, mp4=True)