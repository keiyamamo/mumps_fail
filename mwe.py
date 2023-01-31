import petsc4py
petsc4py.init(['-on_error_attach_debugger'])
from petsc4py import PETSc
from mpi4py import MPI


# Suppose we have a matrix A and a vector b on the cluster 
viewer_A = PETSc.Viewer().createBinary('/cluster/projects/nn9249k/kei/failed_matrix_with_mumps/matrix-A.dat', 'r')
viewer_b = PETSc.Viewer().createBinary('/cluster/projects/nn9249k/kei/failed_matrix_with_mumps/vector-B.dat', 'r')

print(f"Start loading: {MPI.COMM_WORLD.rank}", flush=True)
A = PETSc.Mat().load(viewer_A)
b = PETSc.Vec().load(viewer_b)
print(f"Load complete: {MPI.COMM_WORLD.rank}", flush=True)

A.view()

# ksp = PETSc.KSP().create()
# ksp.setType('preonly')
# pc = ksp.getPC()
# pc.setType('lu')
# pc.setFactorSolverType('mumps') # Default value "petsc" causes diverging solve
# ksp.setOperators(A)


# uh = b.copy()

# ksp.setUp()
# print(f"Set up complete: {MPI.COMM_WORLD.rank}", flush=True)
# ksp.solve(b, uh)

# print(f"Local solution {uh.array}")
