program main
	use cudafor
	use kernel
	use mpimod
	use cudamod
	use MPI
	
	
	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num,final_result
	integer :: i
	integer :: rank, num_procs ,gpu_rank ,num_gpus
	integer :: n, host_n, gpu_n
	integer :: remainder,remainder_gpu
	integer :: istat, ierr, GPU
	integer, dimension(:),allocatable :: gpu_each_n
	
	
	integer, dimension(:),allocatable :: gpu_devices
	! Host input vectors
	real(8),dimension(:,:),allocatable :: h_a
	real(8),dimension(:,:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:,:),allocatable :: h_c
	
	! Device input vectors
	real(8),device,dimension(:),allocatable :: d_a
	real(8),device,dimension(:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:),allocatable :: d_c
	
	type(cudadeviceprop):: prop
	
	
	! MPI Initialize
	call mpiinit(rank, num_procs)
	print *, 'rank: ', rank+1, '/', num_procs
	! GPU Initialize
	call gpu_init(num_gpus)
	allocate(gpu_devices(num_gpus))
	allocate(gpu_each_n(num_gpus))
	! Size of vectors
	n = 3200000
	gpu_n = n / num_gpus
	
	remainder = mod(n,num_procs)
	if (remainder .eq. 0) then
		host_n = n/num_procs
		print *, 'host_n = ', host_n
	else
		if (rank < remainder) then
			host_n = floor(real(n)/real(num_procs))+1.0d0
		else
			host_n = floor(real(n)/real(num_procs))
		end if
		print *, 'rank = ', rank,',host_n = ', host_n
	end if
	
	
	
	remainder_gpu = mod(host_n,num_gpus)
	print *, 'remainder_gpu = ',remainder_gpu
	if (remainder_gpu .eq. 0) then
		gpu_n = host_n/num_gpus
		print *, 'gpu_n = ', gpu_n
	else
		gpu_n = host_n/num_gpus +1
		do i = 1,num_gpus
			if (i .le. remainder_gpu) then
				gpu_each_n(i) = real(host_n/num_gpus)+1
			else
				gpu_each_n(i) = real(host_n/num_gpus)
			end if
		end do
		print *, 'gpu_each_n = ', gpu_each_n
		print *, 'gpu_n = ', gpu_n
	end if
	
	do i = 1, num_gpus
        gpu_devices(i) = i - 1
    end do

	! Allocate memory for each vector on host
	allocate(h_a(num_gpus, gpu_n))
	allocate(h_b(num_gpus, gpu_n))
	allocate(h_c(num_gpus, gpu_n))
	
	!if (remainder_gpu .eq. 0) then
		do i=1,gpu_n
			do GPU = 1, num_gpus
				h_a(GPU,i) = sin((gpu_n*GPU)+i*1D0)**2
				h_b(GPU,i) = cos((gpu_n*GPU)+i*1D0)**2
			end do
		end do
	!else
	!	do GPU = 1, num_gpus
	!		if (GPU .le. remainder_gpu) then
	!			do i=1,gpu_n
	!				h_a(GPU,i) = sin((gpu_n*GPU)+i*1D0)**2
	!				h_b(GPU,i) = cos((gpu_n*GPU)+i*1D0)**2
	!			end do
	!		else
	!			do i=1,gpu_n-1
	!				h_a(GPU,i) = sin((gpu_n*GPU)+i*1D0)**2
	!				h_b(GPU,i) = cos((gpu_n*GPU)+i*1D0)**2
	!			end do
	!			h_a(GPU,gpu_n) = 0
	!			h_b(GPU,gpu_n) = cos((gpu_n*GPU)+i*1D0)**2
	!		end if
	!	end do
	!end if
	print *, 'h_a h_b'
	
	do GPU = 1, num_gpus
		print *, 'gpu_devices = ',gpu_devices(GPU)
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaGetDeviceProperties(prop,gpu_devices(GPU))
		print *, 'device name = ',trim(prop%name)
	end do
	
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		allocate(d_a(gpu_n))
		allocate(d_b(gpu_n))
		allocate(d_c(gpu_n))
		
		istat=cudaMemcpy(d_a, h_a(GPU,:), gpu_n, cudaMemcpyHostToDevice)
		istat=cudaMemcpy(d_b, h_b(GPU,:), gpu_n, cudaMemcpyHostToDevice)
		
		blockSize = dim3(1024,1,1)
		gridSize = dim3(ceiling(real(gpu_n)/real(blockSize%x)) ,1,1)
		call vecAdd_kernel<<<gridSize, blockSize>>>(gpu_n, d_a, d_b, d_c)
	end do
	
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaMemcpy(h_c(GPU,:), d_c, gpu_n, cudaMemcpyDeviceToHost)
	end do
	
	if (remainder_gpu .ne. 0) then
		do GPU = 1, num_gpus
			if (GPU .gt. remainder_gpu) then
				h_c(GPU,gpu_n) = 0
			end if
		end do
	end if
	
	
	
	sum_num = 0.0;
	do GPU = 1, num_gpus
		do i = 1, gpu_n
			sum_num = sum_num + h_c(GPU,i)
		end do
	end do
	
    call MPI_REDUCE(sum_num, final_result, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
	
	if (rank == 0) then
		print *, 'final result: ', final_result/real(n)
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