program main
	use cudafor
	use kernel
	use cudamod
	implicit none


	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num
	integer :: i
	
	! Size of vectors
	integer :: n
	integer :: gpu_n
	integer :: istat, ierr, GPU
	integer :: num_gpus,devnum
	! Host input vectors
	real(8),dimension(:,:),allocatable :: h_a
	real(8),dimension(:,:),allocatable :: h_b
	!Host output vector
	real(8),dimension(:,:),allocatable :: h_c
	
	! Device input vectors
	integer, dimension(:),allocatable :: gpu_devices
	real(8), device, dimension(:), allocatable :: d_a
	real(8), device, dimension(:), allocatable :: d_b
	!Host output vector
	real(8), device, dimension(:), allocatable :: d_c
	
	integer(kind=cuda_stream_kind), dimension(:), allocatable ::g_stream
	type(cudadeviceprop):: prop
	!type(cudaEvent) :: event
	
	! GPU Initialize
	call gpu_init(num_gpus)

	
	n = 800000
	gpu_n = n / num_gpus
	
	allocate(gpu_devices(num_gpus))
	allocate(g_stream(num_gpus))
	do i = 1, num_gpus
        gpu_devices(i) = i - 1
		istat = cudaStreamCreate(g_stream(GPU))
    end do
	! Allocate memory for each vector on host
	allocate(h_a(num_gpus, gpu_n))
	allocate(h_b(num_gpus, gpu_n))
	allocate(h_c(num_gpus, gpu_n))
			
	
	print*, 'allocate host memory completed'
	! Allocate memory for each vector on GPU
	
	
	
	! Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
	do GPU = 1, num_gpus
		do i = 1, gpu_n
			h_a(GPU,i) = sin((gpu_n*GPU)+i*1D0)**2
			h_b(GPU,i) = cos((gpu_n*GPU)+i*1D0)**2
		end do
	end do
	print*, 'Initialize content of input vectors completed'
	
	! Implicit copy of host vectors to device
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
		istat=cudaMemcpyAsync(d_a, h_a(GPU,:), gpu_n, cudaMemcpyHostToDevice, g_stream(GPU))
		istat=cudaMemcpyAsync(d_b, h_b(GPU,:), gpu_n, cudaMemcpyHostToDevice, g_stream(GPU))
		
		blockSize = dim3(1024,1,1)
		gridSize = dim3(ceiling(real(gpu_n)/real(blockSize%x)) ,1,1)
		!istat = cudaEventSynchronize(event)
		call vecAdd_kernel<<<gridSize, blockSize, g_stream(GPU)>>>(gpu_n, d_a, d_b, d_c)
		istat=cudaMemcpyAsync(h_c(GPU,:), d_c, gpu_n, cudaMemcpyDeviceToHost, g_stream(GPU))
		istat = cudaDeviceSynchronize()
	end do
	
	print*,'calculate completed'
	print*,'copy data of device to host completed'
 
    ! Sum up vector c and print result divided by n, this should equal 1 within error
    sum_num = 0.0;
	do GPU = 1, num_gpus
		do i = 1, gpu_n
			sum_num = sum_num + h_c(GPU,i)
		end do
	end do
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