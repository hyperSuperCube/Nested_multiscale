import numpy as np
import NS as ns
nx = 500
ny = 500

xp = np.linspace(-5,5,nx)
yp = np.linspace(-5,5,ny)

dx = xp[1] - xp[0]
dy = yp[1] - yp[0]

fl = lambda y, t: np.zeros(len(y))
fr = lambda y, t: np.zeros(len(y))

fb = lambda x, t: np.zeros(len(x))
fu = lambda x, t: np.zeros(len(x))

u0 = lambda x, y: np.exp(-15*x**2-15*y**2)

boundary = {
    'type' : ["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet"],
    'value' : [fl, fr, fb, fu]
}

Xp, Yp = np.meshgrid(xp, yp, indexing='ij')
U0 = u0(Xp, Yp)
u = ns.Field(field=U0, field_mesh=(xp, yp), grid_sizing=[dx,dy], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='y_surf', do_interp=True)

cx = -1
cy = 1
T = 0.2
dt = 0.0001
t = 0
while t < T:
    print(t)
    u = u + u.laplace_2()*(dt)
    t += dt

u.plot_field("u1", "X", "Y", "Z")
