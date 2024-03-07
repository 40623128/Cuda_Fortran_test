program main
	use cudafor
	use kernel
	use dataTransfer
	type(dim3) :: blockSize, gridSize ,cube_size
	real(8) :: sum_num
	integer :: h, i, j, k, l, element, error

	! Size of vectors
	integer :: n_x, n_y, n_z
	integer:: size_cube
	integer:: total_element
	real(8) :: t1 ,t2
	! Host input vectors
	real(8),dimension(:,:,:),allocatable :: h_a
	real(8),dimension(:),allocatable :: h_a_1d
	real(8),dimension(:,:,:),allocatable :: h_b
	real(8),dimension(:),allocatable :: h_b_1d
	!Host output vector
	real(8),dimension(:,:,:),allocatable :: h_c
	real(8),dimension(:),allocatable :: h_c_1d
	
	! Device input vectors
	real(8),device,dimension(:,:,:),allocatable :: d_a
	real(8),device,dimension(:,:,:),allocatable :: d_b
	!Host output vector
	real(8),device,dimension(:,:,:),allocatable :: d_c
	
	real(8),device,dimension(:),allocatable :: d_a_1d
	real(8),device,dimension(:),allocatable :: d_b_1d
	!Host output vector
	real(8),device,dimension(:),allocatable :: d_c_1d
	
	type(cudaEvent) :: start_event, end_event
	real :: elapsedTime,shortTime
	integer:: best_x, best_y, best_z
	
	n = 1000
	n_x = 24
	n_y = 24
	n_z = 24
	total_element = n_x*n_y*n_z
	
	size_cube = n_x*n_y*n_z*sizeof(real(8))
	print *, 'n_x => ',n_x
	print *, 'n_y => ',n_y
	print *, 'n_z => ',n_z
	
	! Allocate memory for each vector on host
	allocate(h_a(n_x, n_y, n_z))
	allocate(h_b(n_x, n_y, n_z))
	allocate(h_c(n_x, n_y, n_z))
	!allocate(h_c(n, n_x, n_y, n_z, 5))
	allocate(h_a_1d(total_element))
	allocate(h_b_1d(total_element))
	allocate(h_c_1d(total_element))
	
	! Allocate memory for each vector on device
	allocate(d_a(n_x,n_y,n_z))
	allocate(d_b(n_x,n_y,n_z))
	allocate(d_c(n_x,n_y,n_z))
	allocate(d_a_1d(n_x*n_y*n_z))
	allocate(d_b_1d(n_x*n_y*n_z))
	allocate(d_c_1d(n_x*n_y*n_z))
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	
	do i = 1, n_x
		do j = 1, n_y
			do k = 1, n_z
				h_a(i,j,k) = cos(i*j*k*1.d0)**2
				h_b(i,j,k) = sin(i*j*k*1.d0)**2
			end do
		end do
	end do
	print *, 'h_a Initialized'

	
	call dataTransfer_real8_3Dto1D_type0(h_a,h_a_1d)
	call dataTransfer_real8_3Dto1D_type0(h_b,h_b_1d)
	
	! start and end event create
	error = cudaEventCreate(start_event)
	error = cudaEventCreate(end_event)
	
	!copy host vectors to device
	error = cudaMemcpy(d_a_1d, h_a_1d, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	error = cudaMemcpy(d_b_1d, h_b_1d, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	error = cudaMemcpy(d_a, h_a, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	error = cudaMemcpy(d_b, h_b, n_x*n_y*n_z, cudaMemcpyHostToDevice)
	print *, 'cuda Memcpy Host To Device'
	print *,'================= 3d start ================='
	shortTime = 10000000
	do ntimes_x = 1,10
		do ntimes_y = 1,10
			do ntimes_z = 1,6
				if ((2**ntimes_x)*(2**ntimes_y)*(2**ntimes_z)> 2**12) then
					continue
				else
					error =  cudaEventRecord(start_event,0)
					blockSize = dim3(2**ntimes_x,2**ntimes_y,2**ntimes_z)
					gridSize = dim3(ceiling(real(n_x)/real(blockSize%x)),&
									ceiling(real(n_y)/real(blockSize%y)),&
									ceiling(real(n_z)/real(blockSize%z)))
					! Execute the kernel
					do i = 1,10000
						call vecAdd_kernel<<<gridSize, blockSize>>>(n_x, n_y, n_z, &
																	d_a, d_b,d_c)
					end do
					error = cudaEventRecord(end_event,0)
					istat = cudaDeviceSynchronize()
					istat = cudaEventElapsedTime(elapsedTime, start_event, end_event)
					print *,'blockSize%x=',2**ntimes_x,'blockSize%y=',2**ntimes_y,'blockSize%z=',2**ntimes_z
					print *,'total_size=',2**ntimes_x*2**ntimes_y*2**ntimes_z,'elapsedTime(ms):',elapsedTime
					if (shortTime > elapsedTime) then
						shortTime = elapsedTime
						best_x =2**ntimes_x
						best_y =2**ntimes_y
						best_z =2**ntimes_z
					end if
					
				end if
			end do
		end do
	end do
	print *,'shortTime=',shortTime
	print *,'best_x y z=',best_x,best_y,best_z
	print *,'total =',best_x*best_y*best_z
	print *,'================= 3d end ================='
	error = cudaMemcpy(h_c, d_c, n_x*n_y*n_z, cudaMemcpyDeviceToHost)
    
	sum_num = 0.0;
	do i = 1, n_x
		do j = 1, n_y
			do k = 1, n_z
				sum_num = sum_num + h_c(i,j,k)
			end do
		end do
	end do
    sum_num = sum_num/(n_x*n_y*n_z)
    print *, 'final result: ', sum_num
	
	
	
	print *,'================= 1d start ================='
	
	do ntimes = 1,10
		error =  cudaEventRecord(start_event,0)
		blockSize = dim3(2**ntimes,1,1)
		gridSize = dim3(ceiling(real(n_x*n_y*n_z)/real(blockSize%x)), 1, 1)
		! Execute the kernel
		do i = 1,10000
			call vecAdd_kernel_1d<<<gridSize, blockSize>>>(n_x, n_y, n_z, &
														d_a_1d, d_b_1d, d_c_1d)
		end do
		error = cudaEventRecord(end_event,0)
		istat = cudaDeviceSynchronize()
		istat = cudaEventElapsedTime(elapsedTime, start_event, end_event)
		print *,'blockSize%x=',2**ntimes,'elapsedTime(ms):',elapsedTime
		    ! copy device array to host
		h_c_1d(:) = 0.d0
		error = cudaMemcpy(h_c_1d, d_c_1d, n_x*n_y*n_z, cudaMemcpyDeviceToHost)
		
		sum_num = 0.0;
		do element = 1, n_x*n_y*n_z
			sum_num = sum_num + h_c_1d(element)
		end do
		sum_num = sum_num/(n_x*n_y*n_z)
		print *, 'final result: ', sum_num
		
	end do
	print *,'=================  1d end  ================='

	
    ! Release device memory
    deallocate(d_a)
    deallocate(d_b)
    deallocate(d_c)
	
    ! Release host memory
    deallocate(h_a)
    deallocate(h_b)
    deallocate(h_c)
 
end program main