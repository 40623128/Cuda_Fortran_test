program main
	use cudafor
	use kernel
	use mpimod
	use cudamod
	use MPI
	
	
	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num,final_result
	integer :: i
	integer :: rank, num_procs ,gpu_rank ,num_gpu
	integer :: n, host_n, gpu_n
	integer :: remainder
	integer :: istat, ierr, GPU
	
	! Host input vectors
	real(8),dimension(:,:),allocatable :: h_a
	real(8),dimension(:,:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:,:),allocatable :: h_c
	
	! Device input vectors
	real(8),device,dimension(:,:),allocatable :: d_a
	real(8),device,dimension(:,:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:,:),allocatable :: d_c
	
	! MPI Initialize
	call mpiinit(rank, num_procs)
	print *, 'rank: ', rank+1, '/', num_procs
	! GPU Initialize
	call gpu_init(num_gpu)
	! Size of vectors
	do GPU = 0,7
		istat=cudaSetDevice(GPU)
	end do
	
	n = 32000
	
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
	
	if (mod(n,num_procs) .eq. 0) then
		remainder = mod(n,num_procs)
		gpu_n = host_n/num_procs
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
	allocate(h_a(num_gpu, host_n/num_gpu))
	allocate(h_b(num_gpu, host_n/num_gpu))
	allocate(h_c(num_gpu, host_n/num_gpu))
	
	! Allocate memory for each vector on GPU
	allocate(d_a(num_gpu, host_n/num_gpu))
	allocate(d_b(num_gpu, host_n/num_gpu))
	allocate(d_c(num_gpu, host_n/num_gpu))
	
	
	if (remainder == 0) then
		print *, 'remainder=', remainder, 'rank=', rank, 'host_n=', host_n
		do i=1,host_n/num_gpu
			do GPU = 0,7
				h_a(GPU+1,i) = sin((i+(rank*num_gpu+GPU)*host_n/num_gpu)*1D0)**2
				h_b(GPU+1,i) = cos((i+(rank*num_gpu+GPU)*host_n/num_gpu)*1D0)**2
			end do
		end do
	!else
	!	print *, 'remainder=', remainder, 'rank=', rank, 'host_n=', host_n
	!	if (rank <remainder) then
	!		do i=1,host_n
	!			h_a(i) = sin((i+rank*host_n)*1D0)**2
	!			h_b(i) = cos((i+rank*host_n)*1D0)**2
	!		end do
	!	else
	!		do i=1,host_n
	!			h_a(i) = sin((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)**2
	!			h_b(i) = cos((i+(rank-remainder)*host_n+remainder*(host_n+1))*1D0)**2
	!		end do
	!	end if
	end if
	print *, 'h_a h_b'
	
	d_a = h_a
	d_b = h_b
	print *, 'd_a(GPU+1,:) = h_a(GPU+1,:)'
	ierr = cudaDeviceSynchronize()
	
	! Number of threads in each thread block
	blockSize = dim3(1024,1,1)
	! Number of thread blocks in grid
	gridSize = dim3(ceiling(real(gpu_n)/real(blockSize%x)) ,1,1)
	print *, 'gridSize'
	! Execute the kernel
	do GPU = 0,0
		istat=cudaSetDevice(GPU)
		print *, 'cudaSetDevice'
		call vecAdd_kernel<<<gridSize, blockSize>>>(gpu_n, d_a, d_b, d_c)
	end do
	print *, 'vecAdd_kernel'
	ierr = cudaDeviceSynchronize()
	
    ! Implicit copy of device array to host
	do GPU = 0,7
		!istat=cudaSetDevice(GPU)
		h_c(GPU,:) = d_c(GPU,:)
	end do
    ierr = cudaDeviceSynchronize()
	print *, 'vecAdd_kernel'
    ! Sum up vector c and print result divided by n, this should equal 1 within error
    sum_num = 0.0;
    do i=1,host_n
		do GPU = 0,7
			sum_num = sum_num +  h_c(GPU+1,i)
		end do
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