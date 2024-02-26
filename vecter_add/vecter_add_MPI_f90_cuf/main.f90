program main
	use cudafor
	use kernel
	use mpimod
	use MPI
	
	
	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num,final_result
	integer :: i
	integer :: rank, num_procs
	integer :: n,host_n
	integer :: remainder
	
	! Host input vectors
	real(8),dimension(:),allocatable :: h_a
	real(8),dimension(:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:),allocatable :: h_c
	
	! Device input vectors
	real(8),device,dimension(:),allocatable :: d_a
	real(8),device,dimension(:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:),allocatable :: d_c
	
	! MPI Initialize
	call mpiinit(rank, num_procs)
	print *, 'rank: ', rank+1, '/', num_procs
	
	! Size of vectors
	
	n = 100000
	
	if (mod(n,num_procs) .eq. 0) then
		remainder = mod(n,num_procs)
		host_n = n/num_procs
		print *, 'host_n = ', host_n
	else
		remainder = mod(n,num_procs)
		if (rank < remainder) then
			host_n = floor(real(n)/real(num_procs))+1.0d0
		else
			host_n = floor(real(n)/real(num_procs))
		end if
		print *, 'rank = ', rank,',host_n = ', host_n
	end if
	
	! Allocate memory for each vector on host
	allocate(h_a(host_n))
	allocate(h_b(host_n))
	allocate(h_c(host_n))
	
	! Allocate memory for each vector on GPU
	allocate(d_a(host_n))
	allocate(d_b(host_n))
	allocate(d_c(host_n))
	
	if (remainder == 0) then
		print *, 'remainder=', remainder, 'rank=', rank, 'host_n=', host_n
		do i=1,host_n
			h_a(i) = sin((i+rank*host_n)*1D0)*sin((i+rank*host_n)*1D0)
			h_b(i) = cos((i+rank*host_n)*1D0)*cos((i+rank*host_n)*1D0)
		end do
	else
		print *, 'remainder=', remainder, 'rank=', rank, 'host_n=', host_n
		if (rank <remainder) then
			do i=1,host_n
				h_a(i) = sin((i+rank*host_n)*1D0)*sin((i+rank*host_n)*1D0)
				h_b(i) = cos((i+rank*host_n)*1D0)*cos((i+rank*host_n)*1D0)
			end do
		else
			do i=1,host_n
				h_a(i) = sin((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)*sin((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)
				h_b(i) = cos((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)*cos((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)
			end do
		end if
	end if
	
	!do i=1,host_n
	!	h_a(i) = sin(i*1D0)*sin(i*1D0)
	!	h_b(i) = cos(i*1D0)*cos(i*1D0)
	!end do
	
	! Implicit copy of host vectors to device
	d_a = h_a(1:host_n)
	d_b = h_b(1:host_n)
	
	! Number of threads in each thread block
	!blockSize = dim3(1024,1,1)
	blockSize = 1024
	! Number of thread blocks in grid
	gridSize = dim3(ceiling(real(n)/real(blockSize%x)) ,1,1)
	gridSize = ceiling(real(n)/real(blockSize%x))
	! Execute the kernel
    call vecAdd_kernel<<<gridSize, blockSize>>>(host_n, d_a, d_b, d_c)
 
    ! Implicit copy of device array to host
    h_c = d_c(1:host_n)
 
    ! Sum up vector c and print result divided by n, this should equal 1 within error
    sum_num = 0.0;
    do i=1,host_n
        sum_num = sum_num +  h_c(i)
    end do
	
    call MPI_REDUCE(sum_num, final_result, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
	
	if (rank == 0) then
		print *, 'final result: ', final_result
	end if
 
    ! Release device memory
    deallocate(d_a)
    deallocate(d_b)
    deallocate(d_c)
 
    ! Release host memory
    deallocate(h_a)
    deallocate(h_b)
    deallocate(h_c)
 
end program main