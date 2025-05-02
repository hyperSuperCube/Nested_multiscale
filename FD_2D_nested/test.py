import NS as ns
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
# xp = np.linspace(-np.pi,np.pi,400)
# dx = xp[1] - xp[0]
# # xp = np.array([dx/2-np.pi + k*dx for k in range(499)])
# # print(xp[-1])
# # print(xp[-1]+dx/2)
# # y = np.sin(2*x)
# yp = np.sin(2*xp)
# dyp = 2*np.cos(2*xp)
# fl = lambda t: np.sin(-2*(np.pi))
# fr = lambda t: np.sin(2*(np.pi))
# boundary = {'type' : ["Dirichlet", "Dirichlet"], 'value' : [fl, fr]}

# u = ns.field(field=yp, field_mesh=(xp,), grid_sizing=[dx], coord_idx=[0], order=2, time=0, boundary=boundary, store_loc="x_surf")
# x = u.field_mesh[0]
# dudx = u.grad_x()
# plt.figure(dpi=300)
# plt.plot(xp[1:-1], dyp[1:-1], '-')
# plt.plot(x,dudx)
# plt.show()
# print(la.norm(dyp[1:-1]-dudx))


# xp = np.linspace(-np.pi, np.pi,101)
# yp = np.linspace(-np.pi, np.pi,201)
# dx = xp[1]-xp[0]
# dy = yp[1] - yp[0]
# Yp, Xp = np.meshgrid(xp, yp)




nx = 150
ny = 150

xp = np.linspace(-1,1,nx)
yp = np.linspace(-np.pi,np.pi,ny)

dx = xp[1] - xp[0]
dy = yp[1] - yp[0]

Xp, Yp = np.meshgrid(xp, yp, indexing='ij')

fx = np.cos(2*Xp)*np.sin(2*Yp)


fl = lambda y, t: np.cos(-2)*np.sin(2*(y))
fr = lambda y, t: np.cos(2)*np.sin(2*(y))

fb = lambda x_, t: np.cos(2*(x_))*np.sin(2*(-np.pi))
fu = lambda x_, t: np.cos(2*(x_))*np.sin(2*np.pi)


boundary = {
    'type' : ["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet"],
    'value' : [fl, fr, fb, fu]
}

u = ns.Field(field=fx, field_mesh=(xp, yp), grid_sizing=[dx,dy], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='center', do_interp=True)

u2 = ns.Field(field=fx, field_mesh=(xp, yp), grid_sizing=[dx,dy], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='y_surf', do_interp=True)

u3 = ns.Field(field=fx, field_mesh=(xp, yp), grid_sizing=[dx,dy], 
            coord_idx=[0,1], order=2, time=0,
            boundary=boundary, store_loc='x_surf', do_interp=True)
# grad_u = u.grad_x_2()
grad_u = u.grad_y_2()
grad_u2 = u2.grad_x_2()
grad_u3 = u3.grad_y_2()
# grad_u.plot_field("grady", "X", "Y", "Z")

# new_g = grad_u-grad_u2
# new_g.plot_field("grady", "X", "Y", "Z")

x, y = u.field_mesh
X, Y = np.meshgrid(x, y, indexing='ij')
# true_grad_y = 2*np.cos(2*X)*np.cos(2*Y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, grad_u, cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('F')
# plt.show()
true_grad_x = -2*np.sin(2*X)*np.sin(2*Y)
true_grad_y = 2*np.cos(2*X)*np.cos(2*Y)

print(la.norm((grad_u.field).flatten("F") - true_grad_y.flatten("F"),np.inf))


# nx = 100
# ny = 200

# xp = np.linspace(-1,1,nx)
# yp = np.linspace(-np.pi,np.pi,ny)

# dx = xp[1] - xp[0]
# dy = yp[1] - yp[0]

# Xp, Yp = np.meshgrid(xp, yp, indexing='xy')

# fx = np.cos(2*Xp)*np.sin(3*Yp)

# u = ns.field(field=fx, field_mesh=(yp, xp), grid_sizing=[dy,dx], 
#              coord_idx=[1,0], order=2, time=0, boundary=boundary, 
#              store_loc='y_surf', do_interp=True)
# grad_u = u.grad_x_2()
# # print('FIELD HAS SHAPE:',u.field.shape)
# y, x = u.field_mesh
# X, Y = np.meshgrid(x, y, indexing='xy')

# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, grad_u, cmap='viridis',label="Plot")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('F')
# plt.legend()
# plt.show()
# true_grad = dfx = -2*np.sin(2*X)*np.sin(3*Y)

# print(la.norm(grad_u.flatten("F") - true_grad.flatten("F"),np.inf))
import copy

# A = np.random.rand(3,3,3)
# C = copy.deepcopy(A[1])
# B = A.transpose(1,0,2)
# D = copy.deepcopy(B[:,1,:])
# print(C==D)
# print(np.linspace(0,1,2))
# def transpose_helper(coord_idx, position):
#     # According to the coord_idx generate a anti/permutation that put position index at 0 index position and back
#     id = np.where(coord_idx == position)[0][0]
#     # print(id)
#     if id == 0:
#         return None, None
#     else:
#         perm = np.linspace(0,len(coord_idx)-1,len(coord_idx),dtype=int)
#         perm[:id] += 1
#         perm[id] = 0
#         anti_perm = np.linspace(0,len(coord_idx)-1,len(coord_idx),dtype=int)
#         anti_perm[1:id+1] -= 1
#         anti_perm[0] = id
#         return perm, anti_perm 
    
# transpose_helper(np.array([0,1]), 1)