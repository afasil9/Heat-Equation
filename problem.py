#%%
from mpi4py import MPI
import numpy
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from ufl import SpatialCoordinate, sin, pi, grad, div, variable, diff, dx
from dolfinx.fem import Function, Expression, dirichletbc, form
import numpy as np
from ufl.core.expr import Expr

ti = 0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

n = 8

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.hexahedron)

t = variable(fem.Constant(domain, d_t))
dt = fem.Constant(domain, d_t)

V = functionspace(domain, ("Lagrange", 1))

x = SpatialCoordinate(domain)

def exact(x, t):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * t)

uex = exact(x,t)

# Define boundary condition

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

gdim = domain.geometry.dim
facet_dim = gdim - 1

facets = mesh.locate_entities_boundary(domain, dim=facet_dim,
                                        marker= boundary_marker)


bdofs1 = fem.locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)

u_bc_expr_V = Expression(uex, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc1_ex = dirichletbc(u_bc_V, bdofs1)

# Initial Condition

u_n = fem.Function(V)
uex_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(uex_expr)

# np.linalg.norm(u_n.x.array)
#%%
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = -div(grad(uex)) + diff(uex,t)
rhs = f*v*dt*dx + ufl.inner(u_n, v)*dx

lhs = ufl.inner(grad(u), grad(v)) * dt * dx + ufl.inner(u,v)*dx

a = fem.form(lhs)
L = fem.form(rhs)

A = assemble_matrix(a, bcs=[bc1_ex])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], [[bc1_ex]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc1_ex])

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

uh = fem.Function(V)

for n in range(num_steps):
    # Update Diriclet boundary condition
    t.expression().value += dt.value
    u_bc_V.interpolate(u_bc_expr_V)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc1_ex]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc1_ex])

    # Solve linear problem
    ksp.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array


def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))

print(f"error = {L2_norm(u_n - uex)}")
