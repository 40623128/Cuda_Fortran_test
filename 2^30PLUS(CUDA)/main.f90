program mpicuda
use cudamod
use cudafor
implicit none

integer :: i, rank,gpu_rank
integer, parameter :: n = 2**30
integer :: result
integer, dimension(1),device :: d_result

call gpu_init(rank)

! Allocate memory on device
call cudaMalloc(d_result, n * size(real))

call gpu_add_kern<<<1, 256>>>(d_result, n)
! Copy the result back to host
result = d_result(1)

! Deallocate memory on device

print *, "Result:", result

end program