import numpy as np
from mpi4py import MPI


def left_sum(f, a, b, n):
    riemann_sum = 0
    for x in np.linspace(a, b, n, endpoint=False):
        riemann_sum += f(x)
    delta = (b - a) / n
    riemann_sum *= delta
    return riemann_sum


def right_sum(f, a, b, n):
    riemann_sum = 0
    for x in np.linspace(b, a, n, endpoint=False):
        riemann_sum += f(x)
    delta = (b - a) / n
    riemann_sum *= delta
    return riemann_sum


comm = MPI.COMM_WORLD
i = comm.Get_rank()
n_procs = comm.Get_size()

n = 20
a = 0
b = np.pi
ai = (b - a) * (i / n_procs)
bi = b * (i + 1) / n_procs

def f(x):
    return np.sin(x)

left_Riemann_sum_i = left_sum(f, ai, bi, n)
right_Riemann_sum_i = right_sum(f, ai, bi, n)
print('Process %d has Left Riemann Sum = %.7f & Right Riemann Sum = %.7f'%
    (i, left_Riemann_sum_i, right_Riemann_sum_i))

left_Riemann_sum = comm.reduce(left_Riemann_sum_i, op=MPI.SUM, root=0)
right_Riemann_sum = comm.reduce(right_Riemann_sum_i, op=MPI.SUM, root=0)

if i==0:
    print('Process %d got the sums!\nLeft Riemann Sum = %.7f\nRight Riemann Sum = %.7f'%
        (i, left_Riemann_sum, right_Riemann_sum))
