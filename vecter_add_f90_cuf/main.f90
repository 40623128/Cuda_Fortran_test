program main
	use cudafor
	use kernel
	
	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num
	integer :: i
	
	! Size of vectors
	integer :: n = 100000
	
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
	
	! Allocate memory for each vector on host
	allocate(h_a(n))
	allocate(h_b(n))
	allocate(h_c(n))
	
	! Allocate memory for each vector on GPU
	allocate(d_a(n))
	allocate(d_b(n))
	allocate(d_c(n))
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	do i=1,n
		h_a(i) = sin(i*1D0)*sin(i*1D0)
		h_b(i) = cos(i*1D0)*cos(i*1D0)
	end do
	
	! Implicit copy of host vectors to device
	d_a = h_a(1:n)
	d_b = h_b(1:n)
	
	! Number of threads in each thread block
	!blockSize = dim3(1024,1,1)
	blockSize = 1024
	! Number of thread blocks in grid
	gridSize = dim3(ceiling(real(n)/real(blockSize%x)) ,1,1)
	gridSize = ceiling(real(n)/real(blockSize%x))
	! Execute the kernel
    call vecAdd_kernel<<<gridSize, blockSize>>>(n, d_a, d_b, d_c)
 
    ! Implicit copy of device array to host
    h_c = d_c(1:n)
 
    ! Sum up vector c and print result divided by n, this should equal 1 within error
    sum_num = 0.0;
    do i=1,n
        sum_num = sum_num +  h_c(i)
    enddo
    sum_num = sum_num/real(n)
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