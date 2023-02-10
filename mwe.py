import petsc4py
# petsc4py.init(['-info'])
from petsc4py import PETSc
from mpi4py import MPI


# If =< 0, diagnostic printing, statistics, and error messages are suppressed.
# petsc_options = {"mat_mumps_icntl_2": 1}
# level of printing for error, warning, and diagnostic messages. 
# 0 =<: no printing, 1: print error messages, 2: print error and warning messages, 
# 3: print error, warning, and diagnostic messages, 4: print error, warning,and all information messages
# petsc_options = {"mat_mumps_icntl_4": 1}
petsc_options = {"mat_mumps_icntl_14": 400}

# Suppose we have a matrix A and a vector b on the cluster 
viewer_A = PETSc.Viewer().createBinary('/Users/keiyamamoto/Documents/FEniCS_sandbox/failed_matrix_with_mumps/matrix-A.dat', 'r')
viewer_b = PETSc.Viewer().createBinary('/Users/keiyamamoto/Documents/FEniCS_sandbox/failed_matrix_with_mumps/vector-B.dat', 'r')
# viewer_A = PETSc.Viewer().createBinary('/Users/keiyamamoto/Documents/FEniCS_sandbox/mumps_fail_TF_fsi/1/matrix-A.dat', 'r')
# viewer_b = PETSc.Viewer().createBinary('/Users/keiyamamoto/Documents/FEniCS_sandbox/mumps_fail_TF_fsi/1/vector-B.dat', 'r')

print(f"Start loading: {MPI.COMM_WORLD.rank}", flush=True)
A = PETSc.Mat().load(viewer_A)
b = PETSc.Vec().load(viewer_b)
print(f"Load complete: {MPI.COMM_WORLD.rank}", flush=True)

ksp = PETSc.KSP().create()
ksp.setType('preonly')
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps') # Default value "petsc" causes diverging solve
# pc.setFactorSolverType('superlu_dist') # Default value "petsc" causes diverging solve
ksp.setOperators(A)

opts = PETSc.Options()
if petsc_options is not None:
    for k, v in petsc_options.items():
        opts[k] = v

ksp.setFromOptions()


uh = b.copy()
print(f"Start set up: {MPI.COMM_WORLD.rank}", flush=True)
ksp.setUp()
print(f"Set up complete: {MPI.COMM_WORLD.rank}", flush=True)
# ksp.solve(b, uh)

# print(f"Local solution {uh.array}")
