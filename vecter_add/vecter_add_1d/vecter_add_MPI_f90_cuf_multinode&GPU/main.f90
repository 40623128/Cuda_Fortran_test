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
	!====== Host input vectors ======!
	real(8),dimension(:,:),allocatable :: h_a
	real(8),dimension(:,:),allocatable :: h_b
	!====== Host output vectors ======!
	real(8),dimension(:,:),allocatable :: h_c
	
	!====== Device input vectors ======!
	! Device input vectors
	real(8),device,dimension(:),allocatable :: d_a
	real(8),device,dimension(:),allocatable :: d_b
	!====== Device output vectors ======!
	real(8),device,dimension(:),allocatable :: d_c
	
	type(cudadeviceprop):: prop
	
	!====== MPI & GPU Initialize ======!
	call mpiinit(rank, num_procs)
	call gpu_init(num_gpus)
	print *, 'rank: ', rank+1, '/', num_procs, 'GPUs:', num_gpus
	

	!====== Size of vectors ======!
	! n = total size of vectors in all node
	! host_n = total size of vectors in each node
	! gpu_n = total size of vectors in each gpu
	n = 3200000
	
	remainder = mod(n,num_procs)
	if (remainder .eq. 0) then
		host_n = n/num_procs
		!print *, 'host_n = ', host_n
	else
		print *, 'Please make mod(n,num_procs) ==0'
		STOP
		!if (rank < remainder) then
		!	host_n = floor(real(n)/real(num_procs))+1.0d0
		!else
		!	host_n = floor(real(n)/real(num_procs))
		!end if
		!print *, 'rank = ', rank,',host_n = ', host_n
	end if
	
	allocate(gpu_each_n(num_gpus))
	remainder_gpu = mod(host_n,num_gpus)
	!print *, 'remainder_gpu = ',remainder_gpu
	if (remainder_gpu .eq. 0) then
		gpu_n = host_n/num_gpus
		!print *, 'gpu_n = ', gpu_n
	else
		print *, 'Please make mod(host_n,num_gpus) ==0'
		STOP
		!gpu_n = host_n/num_gpus +1
		!do i = 1,num_gpus
		!	if (i .le. remainder_gpu) then
		!		gpu_each_n(i) = real(host_n/num_gpus)+1
		!	else
		!		gpu_each_n(i) = real(host_n/num_gpus)
		!	end if
		!end do
		!print *, 'gpu_each_n = ', gpu_each_n
		!print *, 'gpu_n = ', gpu_n
	end if
	
	
	!====== allocate gpu_devices ======!
	! gpu_devices is use to save Device number for multi-gpu
	! gpu_devices(1) = device 0
	! gpu_devices(n) = device n-1
	allocate(gpu_devices(num_gpus))
	do i = 1, num_gpus
        gpu_devices(i) = i - 1
    end do

	!====== Allocate memory for each vector on host ======!
	allocate(h_a(num_gpus, gpu_n))
	allocate(h_b(num_gpus, gpu_n))
	allocate(h_c(num_gpus, gpu_n))
	
	
	do i=1,gpu_n
		do GPU = 1, num_gpus
			h_a(GPU,i) = sin(rank*host_n+(gpu_n*GPU)+i*1D0)**2
			h_b(GPU,i) = cos(rank*host_n+(gpu_n*GPU)+i*1D0)**2
		end do
	end do
	
	!====== get device name ======!
	do GPU = 1, num_gpus
		!print *, 'gpu_devices = ',gpu_devices(GPU)
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaGetDeviceProperties(prop,gpu_devices(GPU))
		!print *, 'device name = ',trim(prop%name)
	end do
	
	!====== Host To Device Memcpy & start kernel ======!
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
	
	!====== Device To Host Memcpy ======!
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
 
	!====== Release device memory ======!
    deallocate(d_a)
    deallocate(d_b)
    deallocate(d_c)
	
	!====== Release host memory ======!
    deallocate(h_a)
    deallocate(h_b)
    deallocate(h_c)
 
end program main