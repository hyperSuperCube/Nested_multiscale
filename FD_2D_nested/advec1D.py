import NS as ns
import numpy as np
nx = 500
xp = np.linspace(-5,5,nx)
dx = xp[1] - xp[0]

fl = lambda t: 0
fr = lambda t: 0

u = lambda x: np.exp(-15*x**2)
d2udx2 = lambda x: 30*(30*x**2-1)*np.exp(-15*x**2)
u0 = u(xp)
u1 = d2udx2(xp)
boundary = {
    'type' : ["Dirichlet", "Dirichlet"],
    'value' : [fl, fr]
}
u = ns.Field(field=u0, field_mesh=(xp,), grid_sizing=[dx], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='x_surf', do_interp=True)
d2u = ns.Field(field=u1, field_mesh=(xp,), grid_sizing=[dx], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='x_surf', do_interp=True)

# d2fdx2 = u.laplace_2()
T = 0.01
dt = 0.0001
t = 0
while t < T:
    print(t)
    u = u + u.laplace_2()*(dt)
    t += dt

# d2fdx2.plot_field("u1", "X", "Y")
# d2u.plot_field("u1", "X", "Y")
u.plot_field("u1", "X", "Y")
