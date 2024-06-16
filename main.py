import taichi as ti  # import taichi

ti.init(arch=ti.gpu)  # initialize taichi,try to use GPU

# simulation's parameters
n_particles = 8192  # number of particles
n_grid = 128  # number of grids　
dx = 1 / n_grid  # grid's width　
inv_dx = 1 / dx  # inverse of grid's width
dt = 2e-4  # time step　
gravity = 9.8  # gravity　
bound = 3  # boundary

the_c = 2.5e-2  # critical compression
the_s = 5.0e-3  # critical stretch
xi = 5  # hardening coefficient
p_rho = 4e2  # particle's density　
E = 1.4e5  # Young's modulus　
nu = 0.45  # Poisson's ratio　
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters　

p_vol = (dx * 0.5) ** 2  # particle's volume　
p_mass = p_vol * p_rho  # particle's mass

# particle's parameters
x = ti.Vector.field(n=2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(n=2, dtype=float, shape=n_particles)  # velocity　
C = ti.Matrix.field(n=2, m=2, dtype=float, shape=n_particles)  # Affine matrix　
F = ti.Matrix.field(n=2, m=2, dtype=float, shape=n_particles)  # elastic deformation gradient
Fp = ti.Matrix.field(n=2, m=2, dtype=float, shape=n_particles)  # plastic deformation gradient
# J = ti.field(dtype=float, shape=n_particles) #Jacobian determinant
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation　

# grid's mv and mass
grid_v = ti.Vector.field(n=2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_f = ti.Vector.field(n=2, dtype=float, shape=(n_grid, n_grid))


@ti.kernel
def substep():
    # reset grid's velocity and mass
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # Particle to grid: particle's mass and velocity to grid's mass and velocity
    for p in x:
        Xp = x[p] * inv_dx  # particle's position in grid　
        base = int(Xp - 0.5).cast(int)  # the left bottom grid of particle　
        fx = x[p] * inv_dx - base.cast(float)  # the distance between particle and the left bottom grid　
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # weight,use quadratic interpolation　
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]  # weight's derivative　
        mu, la = mu_0, lambda_0  # Lame parameters: with h
        affine = C[p]
        J = 1.0

        #for d in ti.static(range(2)):
        #    new_sig = sig[d, d]
        #    new_sig = ti.min(ti.max(sig[d, d], 1 - the_c), 1 + the_s)  # Plasticity
        #    Jp[p] *= sig[d, d] / new_sig  # Plastic deformation
        #    sig[d, d] = new_sig
        #    J *= new_sig
        #    F[p] = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity

        stress = mu * (F[p] - ti.Matrix.identity(float, 2)) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * ti.log(J)


        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3*3 grids around the particle
            offset = ti.Vector([i, j])  # offset,offset is the index of the grid
            dpos = (offset.cast(float) - fx) * dx  # dpos
            weight = w[i].x * w[j].y  # weight,w[i][0] is the weight of x direction,w[j][1] is the weight of y direction

            grid_m[base + offset] += weight * p_mass  # grid's mass
            grid_v[base + offset] += weight * p_mass * (v[p] + affine @ dpos)

            dweight = ti.Vector.zero(float, 2)
            dweight[0] = inv_dx * dw[i][0] * w[j][1]
            dweight[1] = inv_dx * w[i][0] * dw[j][1]

            force = -p_vol * stress @ dweight  # This is doing Step 6: Add elastic force

            grid_f[base + offset] += p_vol * stress @ F[p].transpose() @ dweight  # This is computing -J * p * F * grad(w)

            grid_v[base + offset] += dt * force

    # compute the grid's velocity and boundary conditions
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # if the weight of the grid is not zero
            grid_v[i, j] /= grid_m[i, j]  # grid's x velocity is the average of all particles' x velocity in the grid
        grid_v[i, j].y -= dt * gravity  # grid's y velocity
        if i < bound and grid_v[
            i, j].x < 0:  # if the index of the grid is smaller than bound and the grid's x velocity is smaller than zero
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[
            i, j].x > 0:  # if the index of the grid is bigger than n_grid - bound and the grid's x velocity is bigger than zero
            grid_v[i, j].x = 0
        if j < bound and grid_v[
            i, j].y < 0:  # if the index of the grid is smaller than bound and the grid's y velocity is smaller than zero
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[
            i, j].y > 0:  # if the index of the grid is bigger than n_grid - bound and the grid's y velocity is bigger than zero
            grid_v[i, j].y = 0

    # Grid to particle: grid's velocity to particle's velocity
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)  # the left bottom grid of particle
        fx = x[p] * inv_dx - base.cast(float)  # the distance between particle and the left bottom grid
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]  # weight,use quadratic interpolation
        dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]  # weight's derivative

        new_v = ti.Vector.zero(ti.f32, 2)  # new velocity
        new_C = ti.Matrix.zero(ti.f32, 2, 2)  # new affine matrix
        new_F = ti.Matrix.zero(float, 2, 2)

        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3*3 grids around the particle
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx  # dpos is the distance between the grid and the particle
            weight = w[i].x * w[j].y

            dweight = ti.Vector.zero(float, 2)
            dweight[0] = inv_dx * dw[i][0] * w[j][1]
            dweight[1] = inv_dx * w[i][0] * dw[j][1]

            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(
                dpos) / dx ** 2  # outer_product is the tensor product of two vectors
            new_F += g_v.outer_product(dweight)

        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]
        F[p] = (ti.Matrix.identity(float, 2) + dt * new_F) @ F[p]  # deformation gradient update


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.3 + 0.4, ti.random() * 0.3 + 0.2]  # randomly initialize the position of particles
        v[i] = [0, 0]  # initialize the velocity of particles
        F[i] = ti.Matrix([[1, 0], [0, 1]])  # initialize the deformation gradient of particles
        C[i] = ti.Matrix.zero(float, 2, 2)
        Jp[i] = 1


init()  # initialize the particles

gui = ti.GUI("MPM_2016")

# simulate 50 steps
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(50):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0xFFFFFF)
    gui.show()