program main
	use cudafor
	use kernel
	use dataTransfer
	type(dim3) :: blockSize, gridSize ,cube_size
	real(8) :: sum_num
	integer :: i,error
	
	! Size of vectors
	integer :: n_x, n_y, n_z, total_element
	integer:: size_cube
	
	!time
	real(8) :: T1, T2, pass_t
	integer:: times, n_time
	
	! Host input vectors
	real(8),dimension(:,:,:),allocatable :: h_a
	real(8),dimension(:,:,:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:,:,:),allocatable :: h_c
	
	real(8),dimension(:),allocatable :: h_a_1d
	real(8),dimension(:),allocatable :: h_b_1d
	real(8),dimension(:),allocatable :: h_c_1d
	
	! Device input vectors
	real(8),device,dimension(:,:,:),allocatable :: d_a
	real(8),device,dimension(:,:,:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:,:,:),allocatable :: d_c
	
	real(8),device,dimension(:),allocatable :: d_a_1d
	real(8),device,dimension(:),allocatable :: d_b_1d
	real(8),device,dimension(:),allocatable :: d_c_1d
	
	n = 20
	n_x = n
	n_y = n
	n_z = n
	total_element = n_x*n_y*n_z
	
	size_cube = n_x*n_y*n_z*sizeof(real(8))
	print *, 'n_x => ',n_x
	print *, 'n_y => ',n_y
	print *, 'n_z => ',n_z
	
	! Allocate memory for each vector on host
	allocate(h_a(n_x, n_y, n_z))
	allocate(h_b(n_x, n_y, n_z))
	allocate(h_c(n_x, n_y, n_z))
	allocate(h_a_1d(total_element))
	allocate(h_b_1d(total_element))
	allocate(h_c_1d(total_element))
	! Allocate memory for each vector on device
	allocate(d_a(n_x, n_y, n_z))
	allocate(d_b(n_x, n_y, n_z))
	allocate(d_c(n_x, n_y, n_z))
	allocate(d_a_1d(total_element))
	allocate(d_b_1d(total_element))
	allocate(d_c_1d(total_element))
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	do i = 1, n_x
		do j = 1, n_y
			do k = 1, n_z
				h_a(i,j,k) = n_x*n_y*n_z
				h_b(i,j,k) = n_x*n_y*n_z
				h_c(i,j,k) = 0.3d0
			end do
		end do
	end do
	
	call dataTransfer_real8_3Dto1D_type0(h_a,h_a_1d)
	call dataTransfer_real8_3Dto1D_type0(h_b,h_b_1d)
	
	pass_t = 0
	n_time = 10
	do times = 1, n_time
		!save the time1
		call cpu_time(T1)
		!copy host vectors to device
		error = cudaMemcpy(d_a_1d, h_a_1d, n_x*n_y*n_z, cudaMemcpyHostToDevice)
		error = cudaMemcpy(d_b_1d, h_b_1d, n_x*n_y*n_z, cudaMemcpyHostToDevice)
		!print *, 'MemcpyHostToDevice'
		
		
		!Number of threads in each thread block
		! blockSize = dim3(x,y,z)
		! x , y <= 1024
		! z <= 64
		! x*y*z <=1024
		blockSize = dim3(64,1,1)
		! Number of thread blocks in grid
		gridSize = dim3(ceiling(real(n_x)/real(blockSize%x)),&
						ceiling(real(n_y)/real(blockSize%y)),&
						ceiling(real(n_z)/real(blockSize%z)))
		! Execute the kernel
		error = cudaFuncSetAttribute(vecAdd_kernel_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)
		call vecAdd_kernel_shared<<<gridSize, blockSize,65536>>>(n, d_a_1d, d_b_1d, d_c_1d)
		!call vecAdd_kernel<<<gridSize, blockSize>>>(n, d_a, d_b, d_c)
		
		!print *, 'vecAdd_kernel'
		error = cudaGetLastError()
		if (error /= 0) then
			print *, 'CUDA kernel launch error:', cudaGetErrorString(error)
			! Additional error handling if needed
		endif
		

		! copy device array to host
		error = cudaMemcpy(h_c_1d, d_c_1d, n_x*n_y*n_z, cudaMemcpyDeviceToHost)
		!print *, 'DeviceToHost'
		
		
		sum_num = 0.0;
		do i = 1, n*n*n
			sum_num = sum_num +  h_c_1d(i)
		end do
		sum_num = sum_num/real(n**3)
		print *, 'final result: ', sum_num
		
		!save the time2
		call cpu_time(T2)
		pass_t = pass_t + T2-T1
		print *, 'time: ',T2-T1
	end do
	print*, 'average time:', pass_t/n_time
	
    ! Release device memory
    deallocate(d_a)
    deallocate(d_b)
    deallocate(d_c)
 
    ! Release host memory
    deallocate(h_a)
    deallocate(h_b)
    deallocate(h_c)
 
end program main