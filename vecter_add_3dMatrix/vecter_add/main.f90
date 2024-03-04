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
				h_a(i,j,k) = sin(i*j*k*1D0)**2
				h_b(i,j,k) = cos(i*j*k*1D0)**2
			end do
		end do
	end do
	
	!copy host vectors to device
	error = cudaMemcpy(d_a(:,:,:), h_a(:,:,:), n_x*n_y*n_z, cudaMemcpyHostToDevice)
	error = cudaMemcpy(d_b(:,:,:), h_b(:,:,:), n_x*n_y*n_z, cudaMemcpyHostToDevice)
	
	!Number of threads in each thread block
	blockSize = dim3(8,8,8)
	! Number of thread blocks in grid
	gridSize = dim3(ceiling(real(n_x)/real(blockSize%x)) ,ceiling(real(n_y)/real(blockSize%y)),ceiling(real(n_z)/real(blockSize%z)))
	! Execute the kernel
	
    call vecAdd_kernel<<<gridSize, blockSize>>>(n, d_a, d_b, d_c)
    ! copy device array to host
	error = cudaMemcpy(h_c(:,:,:), d_c(:,:,:), n_x*n_y*n_z, cudaMemcpyDeviceToHost)
	
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