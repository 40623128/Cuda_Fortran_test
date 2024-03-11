program main
	use cudafor
	use kernel
	
	type(dim3) :: blockSize, gridSize ,cube_size
	real(8) :: sum_num
	integer :: i,error
	
	! Size of vectors
	integer :: n_x, n_y, n_z
	integer:: size_cube
	
	! Host input vectors
	real(8),dimension(:,:,:),allocatable :: h_a
	real(8),dimension(:,:,:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:,:,:),allocatable :: h_c
	
	! Device input vectors
	real(8),device,dimension(:,:,:),allocatable :: d_a
	real(8),device,dimension(:,:,:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:,:,:),allocatable :: d_c
	
	!integer ishap(3)
	!integer,pointer,device :: idev(:,:,:)
	!type(cudaPitchedPtr) :: devPtr
	type(cudaExtent) :: extent
	type(cudaPos) :: offsetPos
	!
	!extent%width = 14
	!extent%height = 14
	!extent%depth = 14
	!offsetPos%x =
	!offsetPos%y =
	!offsetPos%z =
	!error = cudaMalloc3D(devPtr, extent)
	!ishap(1) = devPtr%pitch / 4
	!ishap(2) = n
	!ishap(3) = p
	!
	!call c_f_pointer(devPtr%ptr, idev, ishap)
	!error = cudaMemset3D(devPtr, 9, extent)
	!call c_f_pointer(devPtr%ptr, idev, ishap)
	!wz=idev(1:m,1:n,1:p)
	!write(*,*),'wz =',wz(1,1,:)
	
	
	!cudaMalloc3D
	! 
	n = 24
	n_x = n
	n_y = n
	n_z = n
	
	size_cube = n_x*n_y*n_z*sizeof(real(8))
	print *, 'n_x => ',n_x
	print *, 'n_y => ',n_y
	print *, 'n_z => ',n_z
	
	! Allocate memory for each vector on host
	allocate(h_a(n_x, n_y, n_z))
	allocate(h_b(n_x, n_y, n_z))
	allocate(h_c(n_x, n_y, n_z))
	
	! Allocate memory for each vector on device
	allocate(d_a(n_x, n_y, n_z))
	allocate(d_b(n_x, n_y, n_z))
	allocate(d_c(n_x, n_y, n_z))
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	do i = 1, n_x
		do j = 1, n_y
			do k = 1, n_z
				h_a(i,j,k) = 1.0d0
				h_b(i,j,k) = 1.0d0
				h_c(i,j,k) = 0.3d0
			end do
		end do
	end do
	
	!copy host vectors to device
	error = cudaMemcpy3D(d_a,0,0,h_a,)
	error = cudaMemcpy(d_a, h_a, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	error = cudaMemcpy(d_b, h_b, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	
	
	print *, 'MemcpyHostToDevice'
	
	!Number of threads in each thread block
	! blockSize = dim3(x,y,z)
	! x , y <= 1024
	! z <= 64
	! x*y*z <=1024
	blockSize = dim3(4,4,4)
	! Number of thread blocks in grid
	gridSize = dim3(ceiling(real(n_x)/real(blockSize%x)),&
					ceiling(real(n_y)/real(blockSize%y)),&
					ceiling(real(n_z)/real(blockSize%z)))
	! Execute the kernel
    call vecAdd_kernel_shared<<<gridSize, blockSize>>>(n, d_a, d_b, d_c)
	
	error = cudaGetLastError()
	if (error /= 0) then
		print *, 'CUDA kernel launch error:', cudaGetErrorString(error)
		! Additional error handling if needed
	endif
	
	
	print *, 'vecAdd_kernel'
    ! copy device array to host
	
	error = cudaMemcpy(h_c, d_c, n_x*n_y*n_z, cudaMemcpyDeviceToHost)

	
	print *, 'DeviceToHost'
	
    sum_num = 0.0;
	do i = 1, n_x
		do j = 1, n_y
			do k = 1, n_z
				sum_num = sum_num +  h_c(i,j,k)
			end do
		end do
	end do
    sum_num = sum_num/real(n**3)
    print *, 'final result: ', sum_num
 
    ! Release device memory
    deallocate(d_a)
    deallocate(d_b)
    deallocate(d_c)
 
    ! Release host memory
    deallocate(h_a)
    deallocate(h_b)
    deallocate(h_c)
 
end program main