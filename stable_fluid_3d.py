import taichi as ti
import numpy as np
import time



ti.init(arch=ti.cuda, debug=False)


n = 256
dt = 0.03
dx = 1/n
# dx = 1.0

# 1, 2, 3
RK = 3

#
enable_BFECC = True

#
enable_clipping = True


rho = 1
jacobi_iters = 6


colors = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_colors = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_new_colors = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))

velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
new_new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))

pressures = ti.field(dtype=ti.f32, shape=(n, n, n))
new_pressures = ti.field(dtype=ti.f32, shape=(n, n, n))

divergences = ti.field(dtype=ti.f32, shape=(n, n, n))

pn_max = 10000
pn_current = 0
rate = 10
particles = ti.Vector.field(3, dtype=ti.f32, shape=pn_max)

source_center = ti.Vector([0.5, 0.9, 0.5])
source_radius = 0.05
source_velocity = ti.Vector([0.0, -0.05, 0.0])
# source_velocity = ti.Vector([0.0, 0.0, -0.05])

# screen center. The simulation area is (0, 0) to (1, 1)
# center = ti.Vector([0.5, 0.5, 0.0])

# cell center
stagger = ti.Vector([0.5, 0.5, 0.5])


@ti.func
def I(i, j, k):
	return ti.Vector([i, j, k])


# @ti.func
# def vel(p):
# 	# rotation
# 	# return ti.Vector([p.y-center.y, center.x-p.x])
# 	return sample_bilinear(velocities, p)


@ti.kernel
def init_color_field():
	# random
	for i in ti.grouped(colors):
		# colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
		colors[i] = ti.Vector([0.0, 0.0, 0.0])
		# colors[i] = ti.Vector([1.0, 1.0, 1.0])


@ti.kernel
def init_velocity_field():
	# rotation
	for i in ti.grouped(velocities):
	# 	p = (i + stagger) * dx
	# 	d = p - center
	# 	if d.norm_sqr() < 0.2:
	# 		velocities[i] = ti.Vector([p.y-center.y, center.x-p.x])
		velocities[i] = ti.Vector([0.0, 0.0, 0.0])
		# velocities[i] = ti.Vector([0.0, -1.0, 0.0])
		
@ti.kernel
def init_particles():
	for i in ti.grouped(particles):
		r = ti.sqrt(ti.random()) * source_radius
		a = np.pi * ti.random() * 2
		b = np.pi * ti.random()
		particles[i] = ti.Vector([r * ti.cos(a) * ti.sin(b), r * ti.sin(a) * ti.sin(b), r * ti.cos(b)]) + source_center
		# particles[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
		
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


# @ti.kernel
# def apply_force(velocities:ti.template(), colors:ti.template(), pre_mouse_pos:ti.types.ndarray(ndim=1), cur_mouse_pos:ti.types.ndarray(ndim=1)):

# 	p = ti.Vector([cur_mouse_pos[0], cur_mouse_pos[1]])
# 	pre_p = ti.Vector([pre_mouse_pos[0], pre_mouse_pos[1]])

# 	dp = p - pre_p
# 	dp = dp / max(1e-5, dp.norm())

# 	color = (ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.7 + ti.Vector([0.1, 0.1, 0.1]) * 0.3)

# 	for i, j in velocities:


# 		d2 = (ti.Vector([(i+stagger.x)*dx, (j+stagger.y)*dx]) - p).norm_sqr()

# 		radius = 0.0001
# 		velocities[i, j] = velocities[i, j] + dp * dt * ti.exp(-d2/radius) * 40


# 		if dp.norm() > 0.5:
# 			colors[i, j] = colors[i, j] + ti.exp(-d2 * (4 / (1 / 15)**2)) * color

@ti.kernel
def apply_force_at_point(velocities:ti.template(), colors:ti.template(), pos:ti.template(), r: ti.template(), force:ti.template()):

	dp = force + (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) * 0.001
	# dp = dp / max(1e-5, dp.norm())

	color = (ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.7 + ti.Vector([0.1, 0.1, 0.1]) * 0.3)

	for i, j, k in velocities:


		d2 = (ti.Vector([(i+stagger.x)*dx, (j+stagger.y)*dx, (k+stagger.y)*dx]) - pos).norm_sqr()

		radius = 0.5 * r
		velocities[i, j, k] = velocities[i, j, k] + dp * dt * ti.exp(-d2/radius) * 40
		# if velocities[i, j, k].norm() > 0.00001:
		# 	print(i, j, k, velocities[i, j, k].norm())


		# if dp.norm() > 0.5:
		# 	colors[i, j, k] = colors[i, j, k] + ti.exp(-d2 * (4 / (1 / 15)**2)) * color


@ti.kernel
def decay_color(colors:ti.template()):
	for i in ti.grouped(colors):
		colors[i] = colors[i] * 0.99


@ti.kernel
def update_particles(pn_current:ti.i32):
	for i in range(pn_current):
		new_v = sample_trilinear(velocities, particles[i])
		particles[i] = particles[i] + new_v * dt 



init_color_field()
init_velocity_field()
init_particles()


# gui = ti.GUI("Fluid 2D", (n, n))


# result_dir = "./result"
# video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

# pre_mouse_pos = None
# cur_mouse_pos = None


window = ti.ui.Window("Fluid 3D", (800, 800), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

camera.position(0.5, 0.5, 3)
camera.lookat(0.5, 0.5, 0.5)
scene.set_camera(camera)



while window.running:

	advect(velocities, velocities, new_velocities, new_new_velocities, dt)
	advect(velocities, colors, new_colors, new_new_colors, dt)
	velocities, new_velocities = new_velocities, velocities
	colors, new_colors = new_colors, colors


	# gui.get_event(ti.GUI.PRESS)

	# if gui.is_pressed(ti.GUI.LMB):
	# 	pre_mouse_pos = cur_mouse_pos
	# 	cur_mouse_pos = np.array(gui.get_cursor_pos(), dtype=np.float32)
	# 	if pre_mouse_pos is None:
	# 		pre_mouse_pos = cur_mouse_pos
	# 	apply_force(velocities, colors, pre_mouse_pos, cur_mouse_pos)
	# else:
	# 	pre_mouse_pos = cur_mouse_pos = None
	
	apply_force_at_point(velocities, colors, source_center, source_radius, source_velocity)


	decay_color(colors)

	solve_divergence(velocities, divergences)


	for i in range(jacobi_iters):
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	projection(velocities, pressures)
	
	if pn_current < pn_max:
		pn_current += rate
	update_particles(pn_current)



	# gui.set_image(colors)
	# gui.circles(particles.to_numpy()[:pn_current], radius=1.2, color=0x3399FF)

	# gui.text(content=f'RK {RK}', pos=(0, 0.98), color=0xFFFFFF)
	# if enable_BFECC: 
	# 	gui.text(content=f'BFECC', pos=(0, 0.94), color=0xFFFFFF)
	# 	if enable_clipping: 
	# 		gui.text(content=f'Clipped', pos=(0, 0.90), color=0xFFFFFF)
	
	# gui.show()

	camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.LMB)
	scene.set_camera(camera)

	scene.point_light(pos=(0, 5, 2), color=(1, 1, 1))
	scene.ambient_light((0.5, 0.5, 0.5))

	scene.particles(particles, radius=0.005, color=(1.0, 0.0, 0.0), index_count=pn_current)
	
	canvas.scene(scene)

	window.show()

	# video_manager.write_frame(colors.to_numpy())

# video_manager.make/_video(gif=True, mp4=True)