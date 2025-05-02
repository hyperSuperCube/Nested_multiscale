import NS as ns
import numpy as np
from matplotlib import pyplot as plt
nx = 11
ny = 11

xp = np.linspace(0,1,nx)
yp = np.linspace(0,1,ny)
dx = xp[1] - xp[0]
dy = yp[1] - yp[0]

X, Y = np.meshgrid(xp, yp, indexing='ij')

fxy = lambda x, y: y*np.exp(-5*x**2)*np.cos(3*x*y)
bl = lambda y, t: y 
br = lambda y, t: y*np.exp(-5)*np.cos(3*y)
bb = lambda x, t: x*0
bu = lambda x, t: np.exp(-5*x**2)*np.cos(3*x)


xl = lambda y, t: 0*y
xr = lambda y, t: np.ones(len(y))
xb = lambda x, t: x
xu = lambda x, t: x

yl = lambda y, t: y
yr = lambda y, t: y
yb = lambda x, t: x*0
yu = lambda x, t: np.ones(len(x))

boundary = {
    'type' : ["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet"],
    'value' : [bl, br, bb, bu]
}

boundaryX = {
    'type' : ["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet"],
    'value' : [xl, xr, xb, xu]
}

boundaryY = {
    'type' : ["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet"],
    'value' : [yl, yr, yb, yu]
}

Fxy = fxy(X, Y)
field = ns.Field(Fxy, [xp, yp], [dx, dy], [0,1], 2, 0, boundary, "center", True)

Xfield = ns.Field(X, [xp, yp], [dx, dy], [0,1], 2, 0, boundaryX, "center", True)

xfield_xmesh, xfield_ymesh = Xfield.field_mesh

Bl = bl(xfield_ymesh,0)
Br = br(xfield_ymesh,0)

Bb = bb(xfield_xmesh,0)
Bu = bu(xfield_xmesh,0)

X_xmesh, X_ymesh = np.meshgrid(xfield_xmesh, xfield_ymesh, indexing='ij')

interpFxy = ns.Field.interpolator_general(field.field, np.array([0,1]), [Bl, Br, Bb, Bu], "c2x")
interpX = ns.Field.interpolator_general(X_xmesh, np.array([0,1]), [0, 1, Xfield.field_mesh[0], Xfield.field_mesh[0]], "c2x")
interpY = ns.Field.interpolator_general(X_ymesh, np.array([0,1]), [Xfield.field_mesh[1], Xfield.field_mesh[1], 0, 1], "c2x")
# print(interpX.shape)
a = 0
b = 1

plt.figure(dpi=300)
for yi in yp:
    plt.plot([yi, yi], [a, b], color='gray', lw=0.8)
for xi in xp:
    plt.plot([a, b], [xi, xi], color='gray', lw=0.8)
plt.scatter(interpX,interpY, color='red', s = 5)
plt.scatter(X_xmesh, X_ymesh, color='green', s = 5)
plt.show()

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(interpX, interpY, interpFxy, cmap='viridis', label="interp")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()


# interField = ns.Field(interpFxy, [xp, yp], [dx, dy], [0,1], 2, 0, boundary, "center", False)


# field.plot_field("X", "Y", "Z")