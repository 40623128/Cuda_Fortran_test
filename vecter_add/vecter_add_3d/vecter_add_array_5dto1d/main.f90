program main
	use cudafor
	use kernel
	
	type(dim3) :: blockSize, gridSize ,cube_size
	real(8) :: sum_num
	integer :: h, i, j, k, l, element, error
	
	! Size of vectors
	integer :: n_x, n_y, n_z
	integer:: size_cube
	integer:: total_element
	real(8) :: t1 ,t2
	! Host input vectors
	real(8),dimension(:,:,:,:,:),allocatable :: h_a
	real(8),dimension(:),allocatable :: h_a_1d
	real(8),dimension(:),allocatable :: h_a_1d2
	!Host output vector
	real(8),dimension(:,:,:,:,:),allocatable :: h_c
	
	! Device input vectors
	!real(8),device,dimension(:),allocatable :: d_a
	!Host output vector
	!real(8),device,dimension(:),allocatable :: d_c
	
	! 
	n = 1000
	n_x = 24
	n_y = 24
	n_z = 24
	total_element = n*n_x*n_y*n_z*5
	
	size_cube = n_x*n_y*n_z*sizeof(real(8))
	print *, 'n_x => ',n_x
	print *, 'n_y => ',n_y
	print *, 'n_z => ',n_z
	
	! Allocate memory for each vector on host
	allocate(h_a(n, n_x, n_y, n_z, 5))
	allocate(h_c(n, n_x, n_y, n_z, 5))
	allocate(h_a_1d(total_element))
	allocate(h_a_1d2(total_element))
	
	! Allocate memory for each vector on device
	!allocate(d_a(n, n_x, n_y, n_z, 5))
	!allocate(d_c(n, n_x, n_y, n_z, 5))
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	
	do h = 1, n
		do i = 1, n_x
			do j = 1, n_y
				do k = 1, n_z
					do l = 1, 5
					h_a(h,i,j,k,l) = (h-1)*n_x*n_y*n_z*5 &
									+(i-1)*n_y*n_z*5 &
									+(j-1)*n_z*5 &
									+(k-1)*5 &
									+l
					end do
				end do
			end do
		end do
	end do
	print *, 'h_a Initialized'

	
	call cpu_time(t1)
	do h = 1, n
		do i = 1, n_x
			do j = 1, n_y
				do k = 1, n_z
					do l = 1, 5
					element = (h-1)*n_x*n_y*n_z*5 &
							+ (i-1)*n_y*n_z*5 &
							+ (j-1)*n_z*5 &
							+ (k-1)*5 &
							+ l
					h_a_1d(element) = h_a(h,i,j,k,l)
					end do
				end do
			end do
		end do
	end do
	call cpu_time(t2)
	print *, '5 tier do loop used time:',t2-t1
	print *, 'h_a_1d Initialized'
	
	call cpu_time(t1)
	do element = 1, total_element
		h = (element-1)/(5*n_z*n_y*n_x)+1
		i = mod((element-1)/(5*n_z*n_y), n_x) + 1
		j = mod((element-1)/(5*n_z), n_y) + 1
		k = mod((element-1)/5, n_z) + 1
		l = mod(element-1, 5) + 1
		h_a_1d2(element) = h_a(h,i,j,k,l)
	end do
	call cpu_time(t2)
	print *, '1 tier do use time:',t2-t1
	
	
	print *, 'h_a_1d2 Initialized'
	do element = 1, n*n_x*n_y*n_z*5
		!print*, 'element =',element
		if (h_a_1d2(element) .ne. h_a_1d(element)) then
			print*, 'element =',element
		end if
	end do
	
	!!copy host vectors to device
	!error = cudaMemcpy(d_a, h_a, n*n_x*n_y*n_z*5, cudaMemcpyHostToDevice)
	!!error = cudaMemcpy(d_b, h_b, n*n_x*n_y*n_z*5, cudaMemcpyHostToDevice)
	!print *, 'cuda Memcpy Host To Device'
	!
	!
	!print *, 'cuda Memcpy Host To Device'
	!!Number of threads in each thread block
	!! blockSize = dim3(x,y,z)
	!! x , y <= 1024
	!! z <= 64
	!! x*y*z <=1024
	!blockSize = dim3(8,8,8)
	!! Number of thread blocks in grid
	!gridSize = dim3(ceiling(real(n_x)/real(blockSize%x)),&
	!				ceiling(real(n_y)/real(blockSize%y)),&
	!				ceiling(real(n_z)/real(blockSize%z)))
	!! Execute the kernel
	!h = 1
	!print *, 'h =',h
	!call vecAdd_kernel<<<gridSize, blockSize>>>(n_x, n_y, n_z, &
	!											d_a, d_c)
	!print *, 'vecAdd_kernel'
    !! copy device array to host
	!error = cudaMemcpy(h_c, d_c, n*n_x*n_y*n_z*5, cudaMemcpyDeviceToHost)
	!
    !sum_num = 0.0;
	!
	!do h = 1, n
	!	do i = 1, n_x
	!		do j = 1, n_y
	!			do k = 1, n_z
	!				do l = 1, 1
	!					sum_num = sum_num + h_c(h,i,j,k,l)
	!				end do
	!			end do
	!		end do
	!	end do
	!end do
    !sum_num = sum_num/real(n*n_x*n_y*n_z*15.d0)
    !print *, 'final result: ', sum_num
	!
    !! Release device memory
    !deallocate(d_a)
    !!deallocate(d_b)
    !deallocate(d_c)
	!
    !! Release host memory
    !deallocate(h_a)
    !!deallocate(h_b)
    !deallocate(h_c)
 
end program main