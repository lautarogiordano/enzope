from numba import cuda

@cuda.jit(device=True)
def double_lock(mutex, i, j):
    first, second = (i, j) if i < j else (j, i)

    while cuda.atomic.cas(mutex, first, 0, 1) != 0:
        pass
    while cuda.atomic.cas(mutex, second, 0, 1) != 0:
        pass

    cuda.threadfence()


@cuda.jit(device=True)
def double_unlock(mutex, i, j):
    cuda.threadfence()
    cuda.atomic.exch(mutex, j, 0)
    cuda.atomic.exch(mutex, i, 0)
