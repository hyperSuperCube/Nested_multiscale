import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import RectBivariateSpline
from functools import singledispatchmethod
import copy
class Field:
    '''
        The field is assumed to be uniform mesh in each direction
        The field accept dimension from 1D to 3D
        The field accept gradient calculation of the second order and the forth order
        Along the direction where the index increases, the physical coordinate also increases.
        In 1D the only variable is x. In 2D the only variable is x and y.

        !!! NOTE : : First meshing whole geom then dertermine storing location
    '''
    def __init__(self, field : np.array, 
                 field_mesh : tuple, 
                 grid_sizing : list, coord_idx : list, 
                 order : int, time : np.float64, 
                 boundary : dict, 
                 store_loc : int,
                 do_interp : bool):
        self.dim = len(grid_sizing)
        self.grid_sizing = grid_sizing
        '''
            Contains the grid sizing along each dimension
        '''
        self.coord_idx = coord_idx
        '''
            Coordinate index [1, 2, 0] correspounds to physical coordinates
            [x, y, z]. So making derivative wrt. x means making derivative
            wrt the first axis of the field variable. The defualt is [1,2,0].
            Each increament of the 0th index means an increament on z-axis.
            Each increament of the 1st index means an increament on x-axis.
            Each increament of the 2nd index means an increament on y-axis.
        '''
        self.field = field
        self.field_mesh = field_mesh
        '''
            The field mesh is a tuple, containing
            (x_mesh, y_mesh, z_mesh)
            It temporaraily only supports uniform mesh. It meshes the whole geometry, including bc
            Since this container only support rectangle domain
            with uniform meshes
            x_mesh, y_mesh, z_mesh are all 1D array
        
        '''
        self.order = order
        self.time = time
        self.boundary = boundary
        self.store_loc = store_loc
        self.interpolator = None
        field_shape = self.field.shape
        if do_interp:
            # Construct a new mesh and interpolate
            if store_loc == "x_surf":
                self.mesh_size = np.zeros(self.dim,dtype=int)
                for i in range(self.dim):
                    self.mesh_size[coord_idx[i]] = field_shape[coord_idx[i]] - 1
                self.mesh_size[coord_idx[0]] -= 1
                self.interpolate_to_x_surf()
            elif store_loc == "center":
                self.mesh_size = np.zeros(self.dim, dtype=int)
                for i in range(self.dim):
                    self.mesh_size[coord_idx[i]] = field_shape[coord_idx[i]] - 1 
                self.interpolate_to_center()  
            elif store_loc == "y_surf":
                self.mesh_size = np.zeros(self.dim, dtype=int)  
                for i in range(self.dim):
                    self.mesh_size[coord_idx[i]] = field_shape[coord_idx[i]] - 1
                self.mesh_size[coord_idx[1]] -= 1
                self.interpolate_to_y_surf()
        else:
            if self.dim == 1:
                self.mesh_size = np.array([field_shape[0]])
            elif self.dim == 2:
                self.mesh_size = np.zeros(self.dim,dtype=int)
                for i in range(self.dim):
                    self.mesh_size[coord_idx[i]] = field_shape[coord_idx[i]]
    
    @singledispatchmethod
    def __add__(self, other):
        return NotImplemented

    @singledispatchmethod
    def __sub__(self, other):
        return NotImplemented

    @singledispatchmethod
    def __mul__(self,other):
        return NotImplemented
                
    @__mul__.register(float)
    def _(self, flop : float):
        return Field(self.field*flop, self.field_mesh, self.grid_sizing, 
                             self.coord_idx, self.order, self.time, 
                             self.boundary, self.store_loc, False)

    @singledispatchmethod
    def __truediv__(self, other):
        return NotImplemented

    def first_derivative_mat(self):
        def order2_helper(n, d):
            return (np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1))*1/2/d
        def order4_helper(n, d):
            return (8*np.diag(np.ones(n-1),1) + np.diag(-np.ones(n-2),2) - 8*np.diag(np.ones(n-1),-1) + np.diag(np.ones(n-2),-2))*1/12/d
        if self.order == 2:
            if self.dim == 1:
                nx = self.mesh_size[self.coord_idx[0]]
                return order2_helper(nx, self.grid_sizing[self.coord_idx[0]])
            if self.dim == 2:
                nx = self.mesh_size[self.coord_idx[0]]
                ny = self.mesh_size[self.coord_idx[1]]
                return order2_helper(nx, self.grid_sizing[self.coord_idx[0]]),\
                       order2_helper(ny, self.grid_sizing[self.coord_idx[1]])
            if self.dim == 3:
                nx, ny, nz = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]], self.mesh_size[self.coord_idx[2]]
                dx, dy, dz = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]], self.grid_sizing[self.coord_idx[2]]
                return  order2_helper(nx, dx),\
                        order2_helper(ny, dy),\
                        order2_helper(nz, dz)
        if self.order == 4:
            if self.dim == 1:
                nx = self.mesh_size[self.coord_idx[0]]
                return order4_helper(nx, self.grid_sizing[self.coord_idx[0]])
            if self.dim == 2:
                nx, ny = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]]
                dx, dy = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]]
                return  order4_helper(nx, dx),\
                        order4_helper(ny, dy)
            if self.dim == 3:
                nx, ny, nz = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]], self.mesh_size[self.coord_idx[2]]
                dx, dy, dz = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]], self.grid_sizing[self.coord_idx[2]]
                return  order4_helper(nx, dx),\
                        order4_helper(ny, dy),\
                        order4_helper(nz, dz)

    def second_derivative_mat(self):
        ax = self.coord_idx[0]
        if self.order == 2:
            def helper(nx, dx):
                return (
                    np.diag(np.ones(nx-1), 1)
                    - 2*np.eye(nx)
                    + np.diag(np.ones(nx-1), -1)
                ) / dx**2
            # classic three‐point stencil:  [1, -2, 1] / Δx²
            if self.dim == 1:
                nx = self.mesh_size[ax]
                dx = self.grid_sizing[ax]
                return helper(nx,dx)
            if self.dim == 2:
                nx, ny = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]]
                dx, dy = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]]
                D2x = helper(nx, dx)
                D2y = helper(ny, dy)
                return D2x, D2y

            if self.dim == 3:
                nx, ny, nz = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]], self.mesh_size[self.coord_idx[2]]
                dx, dy, dz = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]], self.grid_sizing[self.coord_idx[2]]
                D2x = helper(nx, dx)
                D2y = helper(ny, dy)
                D2z = helper(nz, dz)
                return D2x, D2y, D2z

        if self.order == 4:
            # five‐point 4th‐order stencil: [-1, 16, -30, 16, -1]/(12 Δx²)
            def helper(nx, dx):
                return (
                -np.diag(np.ones(nx-2),  2)
                +16*np.diag(np.ones(nx-1), 1)
                -30*np.eye(nx)
                +16*np.diag(np.ones(nx-1),-1)
                -np.diag(np.ones(nx-2), -2)
                ) / (12*dx**2)

            if self.dim == 1:
                nx = self.mesh_size[ax]
                dx = self.grid_sizing[ax]
                return helper(nx, dx)

            if self.dim == 2:
                nx, ny = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]]
                dx, dy = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]]
                D2x4 = helper(nx, dx)
                D2y4 = helper(ny, dy)
                return D2x4, D2y4
            
            if self.dim == 3:
                nx, ny, nz = self.mesh_size[self.coord_idx[0]], self.mesh_size[self.coord_idx[1]], self.mesh_size[self.coord_idx[2]]
                dx, dy, dz = self.grid_sizing[self.coord_idx[0]], self.grid_sizing[self.coord_idx[1]], self.grid_sizing[self.coord_idx[2]]
                D2x4 = helper(nx, dx)
                D2y4 = helper(ny, dy)
                D2z4 = helper(nz, dz)
                return D2x4, D2y4, D2z4

    def grad_x_2(self):
        '''
            Making derivative wrt. the index at x position
            boundary: :dict
                {
                    boundary_type:  Nuemann / Dirichlet / Robin / periodic
                    boundary_value: function of space and time, for 1D only time
                }
            Sequence follows (in 3D) [-x, +x, -y, +y, -z, +z]
            The resulting data is still interior, the boundary informaiton is added on rhs
        '''
        bc = self.boundary
        if self.dim == 1:
            Dx = self.first_derivative_mat()
            ulb, urb = 0, 0
            if bc['type'][0] == "Dirichlet":
                ulb = self.enforce_dirichlet_2(0)
            if bc['type'][1] == "Dirichlet":
                urb = self.enforce_dirichlet_2(1)
            dfdx = Dx@self.field
            dfdx[0]  -= ulb/2/self.grid_sizing[self.coord_idx[0]]
            dfdx[-1] += urb/2/self.grid_sizing[self.coord_idx[0]]
            # FIX ME The new boundary condition are all convert to Dirichlet !!!
            return Field(dfdx, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
        elif self.dim == 2:
            Dx, Dy = self.first_derivative_mat()
            ulb, urb, ubb, uub = (0,0,0,0)
            if bc['type'][0] == "Dirichlet":
                ulb = self.enforce_dirichlet_2(0)
            if bc['type'][1] == "Dirichlet":
                urb = self.enforce_dirichlet_2(1)
            idx, idy = self.coord_idx[0], self.coord_idx[1]
            dfdx = 0
            if idx == 0:
                dfdx = Dx@self.field
            elif idx == 1:
                dfdx = self.field@(Dx.T)
            if idx == 0 and idy == 1:
                # Get the left boundary at x = 0
                dfdx[0] -= ulb/2/self.grid_sizing[idx]
                # Get the Right boundary at x = 1
                dfdx[-1] += urb/2/self.grid_sizing[idx]
            if idx == 1 and idy == 0:
                dfdx[:,0] -= ulb/2/self.grid_sizing[idx]
                dfdx[:,-1] += urb/2/self.grid_sizing[idx]
            return Field(dfdx, self.field_mesh, self.grid_sizing, 
                             self.coord_idx, self.order, self.time, 
                             self.boundary, self.store_loc, False)

    def grad_y_2(self):
        bc = self.boundary
        if self.dim == 2:
            Dx, Dy = self.first_derivative_mat()
            ulb, urb, ubb, uub = (0,0,0,0)
            if bc['type'][2] == "Dirichlet":
                ubb = self.enforce_dirichlet_2(2)
            if bc['type'][3] == "Dirichlet":
                uub = self.enforce_dirichlet_2(3)
            idx, idy = self.coord_idx[0], self.coord_idx[1]
            dfdy = 0
            if idx == 0:
                dfdy = self.field@(Dy.T)
            elif idx == 1:
                dfdy = Dy@self.field
            if idx == 1 and idy == 0:
                # Get the bottom boundary at y = 0
                dfdy[0] -= ubb/2/self.grid_sizing[idy]
                # Get the up boundary at y = 1
                dfdy[-1] += uub/2/self.grid_sizing[idy]
            if idx == 0 and idy == 1:
                dfdy[:,0] -= ubb/2/self.grid_sizing[idy]
                dfdy[:,-1] += uub/2/self.grid_sizing[idy]
            return Field(dfdy, self.field_mesh, self.grid_sizing, 
                             self.coord_idx, self.order, self.time, 
                             self.boundary, self.store_loc, False)

    def grad_z_2(self):
        pass

    def laplace_2(self):
        """
            To Do : : After calculating the laplace, the boundary is automatically updated 
        """
        bc = self.boundary
        if self.dim == 1:
            D2x = self.second_derivative_mat()
            ubs = []
            for i in range(2):
                if bc['type'][i] == "Dirichlet":
                    ubs.append(self.enforce_dirichlet_2(i))
                elif bc['type'][i] == "Nuemann":
                    pass
            d2fdx2 = D2x@self.field
            d2fdx2[0]  += ubs[0]/(self.grid_sizing[self.coord_idx[0]])**2
            d2fdx2[-1] += ubs[1]/(self.grid_sizing[self.coord_idx[0]])**2
            return Field(d2fdx2, self.field_mesh, self.grid_sizing, 
                             self.coord_idx, self.order, self.time, 
                             self.boundary, self.store_loc, False)
        elif self.dim == 2:
            D2x, D2y = self.second_derivative_mat()
            idx, idy = self.coord_idx[0], self.coord_idx[1]
            ubs = []
            for i in range(4):
                if bc['type'][i] == "Dirichlet":
                    ubs.append(self.enforce_dirichlet_2(i))
                elif bc['type'][i] == "Nuemann":
                    pass
            d2fdx2, d2fdy2 = 0, 0
            if idx == 0:
                d2fdx2 = D2x @ self.field
                d2fdy2 = self.field @ (D2y.T)
                d2fdx2[0]  += ubs[0]/(self.grid_sizing[idx])**2
                d2fdx2[-1] += ubs[1]/(self.grid_sizing[idx])**2

                d2fdy2[:,0]  += ubs[2]/(self.grid_sizing[idy])**2
                d2fdy2[:,-1] += ubs[3]/(self.grid_sizing[idy])**2
            elif idx == 1:
                d2fdx2 = self.field @ (D2x.T)
                d2fdy2 = D2y @ self.field
                d2fdx2[:,0]  += ubs[0]/(self.grid_sizing[idx])**2
                d2fdx2[:,-1] += ubs[1]/(self.grid_sizing[idx])**2

                d2fdy2[0]  += ubs[2]/(self.grid_sizing[idy])**2
                d2fdy2[-1] += ubs[3]/(self.grid_sizing[idy])**2
            return Field(d2fdx2+d2fdy2, self.field_mesh, self.grid_sizing, 
                        self.coord_idx, self.order, self.time, 
                        self.boundary, self.store_loc, False)
        elif self.dim == 3:
            # Calculate d^2u/dx^2+d^2u/dy^2+d^2u/dz^2
            pass

    def solve_possion_field_2(self, rhs_field, tol, method, iter):
        pass
    
    # This is the general purpose interpolator, using convolution to get a new Field instance
    @staticmethod
    def interpolator_general_2(field, coord_idx, bc_vals, map):
        print(map)
        def vector_avg(vec, bm, bp, type):
            '''
                Doing average along the first dimension direction
            '''
            if type == "x2c" or type == "y2c" or type == "z2c":
                new_shape = list(vec.shape)
                new_shape[0] += 1
                new_shape = tuple(new_shape)
                toReturn = np.zeros(new_shape)
                toReturn[0] = (vec[0]+bm)/2
                toReturn[-1] = (vec[-1]+bp)/2
                toReturn[1:-1] = (vec[1:]+vec[:-1])/2
                return toReturn
            elif type == "c2x" or type == "c2y" or type == "c2z":
                return (vec[1:]+vec[:-1])/2
        def transpose_helper(coord_idx, position):
            # According to the coord_idx generate a anti/permutation that put position index at 0 index position and back
            id = np.where(coord_idx == position)[0][0]
            if id == 0:
                tag = -np.ones(len(coord_idx),dtype=int)
                return tag, tag
            else:
                perm = np.linspace(0,len(coord_idx)-1,len(coord_idx),dtype=int)
                perm[:id] += 1
                perm[id] = 0
                anti_perm = np.linspace(0,len(coord_idx)-1,len(coord_idx),dtype=int)
                anti_perm[1:id+1] -= 1
                anti_perm[0] = id
                return perm, anti_perm 
        # Transpose field to the correct position, field is indexed by coord_idx
        if map == "x2c" or map == "c2x":
            perm, anti_perm = transpose_helper(coord_idx, 0)
            if (perm == -np.ones(len(coord_idx),dtype=int)).all():
                return vector_avg(field, bc_vals[0], bc_vals[1], map)
            field = field.transpose(perm)
            return vector_avg(field, bc_vals[0], bc_vals[1], map).transpose(anti_perm)
        elif map == "y2c" or map == "c2y":
            perm, anti_perm = transpose_helper(coord_idx, 1)
            if (perm == -np.ones(len(coord_idx),dtype=int)).all():
                return vector_avg(field, bc_vals[2], bc_vals[3], map)
            field = field.transpose(perm)
            return vector_avg(field, bc_vals[2], bc_vals[3], map).transpose(anti_perm)
        elif map == "z2c" or map == "c2z":
            perm, anti_perm = transpose_helper(coord_idx, 2)
            if (perm == -np.ones(len(coord_idx),dtype=int)).all():
                return vector_avg(field, bc_vals[4], bc_vals[5], map)
            field = field.transpose(perm)
            return vector_avg(field, bc_vals[4], bc_vals[5], map).transpose(anti_perm)
        elif map == "x2y":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"x2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2y")
        elif map == "y2x":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"y2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2x")
        elif map == "x2z":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"x2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2z")
        elif map == "z2x":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"z2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2z")
        elif map == "y2z":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"y2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2z")
        elif map == "z2y":
            temp = Field.interpolator_general(field, coord_idx, bc_vals,"z2c")
            return Field.interpolator_general(temp, coord_idx, bc_vals,"c2y")
        else:
            return None

    ## The following 3 interpolator are mearly used for initilization NOT GENRAL PURPOSE!! (for calling simplicity)
    def interpolate_to_x_surf(self):
        if self.dim == 1:
            self.interpolator = CubicSpline(self.field_mesh[0], self.field, bc_type='natural')
        elif self.dim == 2:
            self.interpolator = RectBivariateSpline(self.field_mesh[0], self.field_mesh[1], self.field)
        else:
            pass # TO DO
        if self.dim == 1:
            x_new = self.field_mesh[0][1:-1]
            self.field = self.interpolator(x_new)
            self.field_mesh = (x_new,)
        elif self.dim == 2:
            idx = self.coord_idx[0]
            idy = self.coord_idx[1]
            hy = self.grid_sizing[idy]
            ny = self.mesh_size[idy]
            new_x = (self.field_mesh[idx])[1:-1]
            y0 = self.field_mesh[idy][0]
            new_y = np.array([y0+hy/2+k*hy for k in range(ny)])
            if idx == 0:
                self.field = self.interpolator(new_x, new_y)
                self.field_mesh = (new_x, new_y)
            else:
                self.field = self.interpolator(new_y, new_x)
                self.field_mesh = (new_y, new_x)
        self.store_loc = 'x_surf'
        
    def interpolate_to_y_surf(self):
        if self.dim == 2:
            self.interpolator = RectBivariateSpline(self.field_mesh[0], self.field_mesh[1], self.field)
        elif self.dim == 3:
            pass # TO DO
        if self.dim == 2:
            idx = self.coord_idx[0]
            idy = self.coord_idx[1]
            hx = self.grid_sizing[idx]
            nx = self.mesh_size[idx]
            new_y = (self.field_mesh[idy])[1:-1]
            x0 = self.field_mesh[idx][0]
            new_x = np.array([x0+hx/2+k*hx for k in range(nx)])
            if idx == 0:
                self.field = self.interpolator(new_x, new_y)
                self.field_mesh = (new_x, new_y)
            else:
                self.field = self.interpolator(new_y, new_x)
                self.field_mesh = (new_y, new_x)
            
        elif self.dim == 3:
            pass
        self.store_loc = "y_surf"
    
    def interpolate_to_z_surf(self):
        pass

    def interpolate_to_center(self):
        if self.dim == 1:
            self.interpolator = CubicSpline(self.field_mesh[0], self.field, bc_type='natural')
        elif self.dim == 2:
            self.interpolator = RectBivariateSpline(self.field_mesh[0], self.field_mesh[1], self.field)
        else:
            pass # TO DO
        if self.dim == 1:
            return self.interpolate_to_x_surf()
        elif self.dim == 2:
            idx = self.coord_idx[0]
            idy = self.coord_idx[1]
            hy = self.grid_sizing[idy]
            hx = self.grid_sizing[idx]
            ny = self.mesh_size[idy]
            nx = self.mesh_size[idx]
            x0 = self.field_mesh[idx][0]
            y0 = self.field_mesh[idy][0]
            new_x = np.array([x0+hx/2+k*hx for k in range(nx)])
            new_y = np.array([y0+hy/2+k*hy for k in range(ny)])
            if idx == 0:
                self.field = self.interpolator(new_x, new_y)
                self.field_mesh = (new_x, new_y)
            else:
                self.field = self.interpolator(new_y, new_x)
                self.field_mesh = (new_y, new_x)
        self.store_loc = 'center'

    def get_boundary_value(self, position):
        """
            This function is used for obtaining the nodal boundary value from 
            Dirichlet / Nuemann / Robin / Periodic
        """
        bc = self.boundary
        if self.dim == 1:
            if bc["type"][position] == "Dirichlet":
                ub_f = bc["value"][position]
                return ub_f(self.time)
            elif bc["type"][position] == "Nuemann":
                # Get the boundary value fron Nuemann boundary
                pass
        if self.dim == 2:
            idx, idy = self.coord_idx[0], self.coord_idx[1]
            x_mesh, y_mesh = self.field_mesh[idx], self.field_mesh[idy]
            if bc["type"][position] == "Dirichlet":
                ub_f = bc["value"][position]
                if position == 0 or position == 1:
                    return ub_f(y_mesh, self.time)
                else:
                    return ub_f(x_mesh, self.time)
            elif bc["type"][position] == "Nuemann":
                pass

    def enforce_dirichlet_2(self, position):
        '''
            This function return the ghost node/line/face value at specified position
            for the second order calculation.
        '''
        if self.dim == 1:  
            ub_f = self.boundary['value'][position]
            if self.store_loc == "center":
                ub_ghost = 2*ub_f(self.time) - self.field[position*(-1)]
                return ub_ghost
            elif self.store_loc == "x_surf":
                ub_ghost = ub_f(self.time)
                return ub_ghost
        elif self.dim == 2:
            idx, idy = self.coord_idx[0], self.coord_idx[1]
            ub_f = self.boundary['value'][position]
            ax = idx if position < 2 else idy
            u_int = np.take(self.field, (-1)*(position%2), axis=ax)
            if self.store_loc == "center":
                ub_ghost = 2*ub_f(self.field_mesh[1-ax], self.time) - u_int
                return ub_ghost
            elif self.store_loc == "x_surf":
                if position == 0 or position == 1:
                    ub_ghost = ub_f(self.field_mesh[1-ax], self.time)
                    return ub_ghost
                elif position == 2 or position == 3:
                    ub_ghost = 2*ub_f(self.field_mesh[1-ax], self.time) - u_int
                    return ub_ghost
            elif self.store_loc == "y_surf":
                if position == 2 or position == 3:
                    ub_ghost = ub_f(self.field_mesh[1-ax], self.time)
                    return ub_ghost
                elif position == 0 or position == 1:
                    ub_ghost = 2*ub_f(self.field_mesh[1-ax], self.time) - u_int
                    return ub_ghost
        elif self.dim == 3:
            pass
            # TO DO
    
    def enfoce_dirichlet4(self, position):
        pass
            
    def enforce_nuemann2(self, position):
        pass

    def enforce_periodic2(self, position):
        pass

    def enforce_robin2(self, position):
        pass

    def enforce_nuemann4(self, position):
        pass

    def enforce_periodic4(self, position):
        pass

    def enforce_robin4(self, position):
        pass

    def time_increament(self,dt):
        self.time += dt

    # Plotting tools 
    def plot_field(self, label, xlabel, ylabel, zlabel=None, dpi=300):
        if self.dim == 1:
            plt.figure(dpi=dpi)
            plt.plot(self.field_mesh[0], self.field, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.show()
        elif self.dim == 2:
            idx, idy = self.coord_idx
            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            tag = 'xy'
            if idx == 0:
                tag = 'ij'
            X, Y = np.meshgrid(self.field_mesh[idx], self.field_mesh[idy], indexing=tag)
            ax.plot_surface(X, Y, self.field, cmap='viridis', label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            plt.legend()
            plt.show()

    def plot_joint(self, field_list : list, overlay : bool, label, xlabel, ylabel, zlabel=None, dpi=300):
        pass

@Field.__mul__.register
def _(self: Field, other_field: Field) -> Field:
    if (self.store_loc == other_field.store_loc):
            """
                For operator overloading, we regard the addee is the correction of the adder
                Thus, the boundary condition of the addee should not affect the boundary of the
                addor
            """
            # Interpolate mesh of other field to same location of the current mesh
            return Field(self.field*other_field.field, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
    else:
        if self.order == 2:
            if self.dim ==1:
                ulb, urb = self.get_boundary_value(0), self.get_boundary_value(1)
                bc_vals = [ulb, urb]
                tgt_loc = self.store_loc
                src_loc = other_field.store_loc
                map = src_loc[0] + "2" + tgt_loc[0]
                new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
                return Field(self.field*new_field_other, self.field_mesh, self.grid_sizing, 
                                self.coord_idx, self.order, self.time, 
                                self.boundary, self.store_loc, False)
            elif self.dim == 2:
                ulb, urb, ubb, uub = self.get_boundary_value(0), self.get_boundary_value(1), self.get_boundary_value(2), self.get_boundary_value(3)
                bc_vals = [ulb, urb, ubb, uub]
                tgt_loc = self.store_loc
                src_loc = other_field.store_loc
                map = src_loc[0] + "2" + tgt_loc[0]
                new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
                return Field(self.field*new_field_other, self.field_mesh, self.grid_sizing, 
                                self.coord_idx, self.order, self.time, 
                                self.boundary, self.store_loc, False)
        
@Field.__add__.register
def _(self: Field, other_field: Field) -> Field:       
    if (self.store_loc == other_field.store_loc):
            """
                For operator overloading, we regard the addee is the correction of the adder
                Thus, the boundary condition of the addee should not affect the boundary of the
                addor
            """
            return Field(self.field+other_field.field, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
    else:
        # Interpolate mesh of other field to same location of the current mesh
        if self.dim ==1:
            ulb, urb = self.get_boundary_value(0), self.get_boundary_value(1)
            bc_vals = [ulb, urb]
            tgt_loc = self.store_loc
            src_loc = other_field.store_loc
            map = src_loc[0] + "2" + tgt_loc[0]
            new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
            return Field(self.field+new_field_other, self.field_mesh, self.grid_sizing, 
                        self.coord_idx, self.order, self.time, 
                        self.boundary, self.store_loc, False)
        elif self.dim == 2:
            ulb, urb, ubb, uub = self.get_boundary_value(0), self.get_boundary_value(1), self.get_boundary_value(2), self.get_boundary_value(3)
            bc_vals = [ulb, urb, ubb, uub]
            tgt_loc = self.store_loc
            src_loc = other_field.store_loc
            map = src_loc[0] + "2" + tgt_loc[0]
            new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
            return Field(self.field+new_field_other, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
        elif self.dim == 3:
            pass

@Field.__sub__.register
def _(self: Field, other_field: Field) -> Field:  
    if (self.store_loc == other_field.store_loc):
            """
                For operator overloading, we regard the addee is the correction of the adder
                Thus, the boundary condition of the addee should not affect the boundary of the
                addor
            """
            return Field(self.field-other_field.field, self.field_mesh, self.grid_sizing, 
                             self.coord_idx, self.order, self.time, 
                             self.boundary, self.store_loc, False)
    else:
        # Interpolate mesh of other field to same location of the current mesh
        if self.dim ==1:
            ulb, urb = self.get_boundary_value(0), self.get_boundary_value(1)
            bc_vals = [ulb, urb]
            tgt_loc = self.store_loc
            src_loc = other_field.store_loc
            map = src_loc[0] + "2" + tgt_loc[0]
            new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
            return Field(self.field-new_field_other, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
        elif self.dim == 2:
            ulb, urb, ubb, uub = self.get_boundary_value(0), self.get_boundary_value(1), self.get_boundary_value(2), self.get_boundary_value(3)
            bc_vals = [ulb, urb, ubb, uub]
            tgt_loc = self.store_loc
            src_loc = other_field.store_loc
            map = src_loc[0] + "2" + tgt_loc[0]
            new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
            return Field(self.field-new_field_other, self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
        elif self.dim == 3:
            pass

@Field.__truediv__.register
def _(self: Field, other_field: Field) -> Field:  
    if (self.store_loc == other_field.store_loc):

            """
                For operator overloading, we regard the addee is the correction of the adder
                Thus, the boundary condition of the addee should not affect the boundary of the
                addor
            """

            return Field(self.field/(other_field.field+1e-14), self.field_mesh, self.grid_sizing, 
                                self.coord_idx, self.order, self.time, 
                                self.boundary, self.store_loc, False)
    else:
        # Interpolate mesh of other field to same location of the current mesh
        if self.dim ==1:
            ulb, urb = self.get_boundary_value(0), self.get_boundary_value(1)
            bc_vals = [ulb, urb]
            tgt_loc = self.store_loc
            src_loc = other_field.store_loc
            map = src_loc[0] + "2" + tgt_loc[0]
            new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
            return Field(self.field/(new_field_other+1e-15), self.field_mesh, self.grid_sizing, 
                            self.coord_idx, self.order, self.time, 
                            self.boundary, self.store_loc, False)
        elif self.dim == 2:
                ulb, urb, ubb, uub = self.get_boundary_value(0), self.get_boundary_value(1), self.get_boundary_value(2), self.get_boundary_value(3)
                bc_vals = [ulb, urb, ubb, uub]
                tgt_loc = self.store_loc
                src_loc = other_field.store_loc
                map = src_loc[0] + "2" + tgt_loc[0]
                new_field_other = Field.interpolator_general_2(other_field. field, self.coord_idx, bc_vals, map)
                return Field(self.field/(new_field_other+1e-14), self.field_mesh, self.grid_sizing, 
                                self.coord_idx, self.order, self.time, 
                                self.boundary, self.store_loc, False)
        elif self.dim == 3:
            pass

