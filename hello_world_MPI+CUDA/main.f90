program mpicuda

use mpimod
use cudamod

implicit none

integer :: i, rank,gpu_rank
call mpiinit(rank)
call usegpu(rank)
call gpu_init(rank)
call finalizempi()
end program