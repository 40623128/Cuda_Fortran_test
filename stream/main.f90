program mpicuda
	use MPI
	use cudafor
	use cudamod
	use gpu_fuc
	use kernel
	implicit none

	integer :: n, i, j, k
	integer, dimension(:),allocatable :: gpu_devices
	integer :: ntimes
	integer::status(MPI_STATUS_SIZE)
	! Timer
	real(8) :: T1, T2, pass_t , total_time_1, total_time_2
	! GPU
	type(dim3) :: blockSize, gridSize
	integer :: GPU, num_gpus, gpu_istat, local_rank
	!type(cudaError) :: cuda_error
	
	integer, parameter :: num_streams = 8
	!integer(kind=cuda_stream_kind), dimension(:,:), allocatable ::g_stream
	integer(kind=cuda_stream_kind), dimension(:), allocatable ::g_stream
	
	! CPU
	integer :: rank_gpu,rank, istat, num_procs, local_comm 
	
	real(8), dimension(:,:), allocatable :: h_a
	real(8), dimension(:,:), allocatable :: h_b
	real(8), dimension(:,:), allocatable :: h_c
	
	real(8), device, dimension(:), allocatable  :: d_a
	real(8), device, dimension(:), allocatable  :: d_b
	real(8), device, dimension(:), allocatable  :: d_c
	
	!GPU init
	call gpu_init(num_gpus)
	num_gpus = 1
	print*, 'num_gpus', num_gpus
	!para init
	n = 0
	
	!test times
	ntimes = 1

	!give every gpu a index
	allocate(gpu_devices(num_gpus))
	allocate(g_stream(num_gpus))
	do GPU = 1, num_gpus
		gpu_devices(GPU) = GPU - 1
		g_stream(GPU) = GPU
	end do
	!allocate(g_stream(GPU)(num_gpus, num_streams))
	!allocate(g_stream(GPU)(num_gpus))
	print*, 'stream create'
	
	
	
	!Gpu stream create
	!
	do GPU = 1, num_gpus
		!gpu_istat = cudaSetDevice(gpu_devices(GPU))
		!gpu_istat = cudaStreamCreate(g_stream(GPU)(GPU))

		gpu_istat = cudaStreamCreate(g_stream(GPU))
	end do
	call deviceSynchronize(gpu_devices)
	
	! Start testing
	! j is n's size (if j = 20 then n = 2^20 ......) 
	! n is data size
	write(*,*) 'Start testing'
	do j = 20, 20
		write(*,*) j
		n = 2**4
		total_time_1 = 0
		total_time_2 = 0
		
		do k = 1, ntimes
			! CPU memory Allocate
			! h_a_send for mpi send
			! h_a_recv for mpi recv
			allocate(h_a(num_gpus,n))
			allocate(h_b(num_gpus,n))
			allocate(h_c(num_gpus,n))
			
			! Generation problen
			! gpu 1 = 1, gpu 2 =2,...
			do GPU = 1, num_gpus
				do i = 1, n
					h_a(GPU,i) = i!sin(i*1D0)**2
					h_b(GPU,i) = i!cos(i*1D0)**2
				end do
			end do
			
			!GPU memory Allocate
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				allocate(d_a(n))
				allocate(d_b(n))
				allocate(d_c(n))
				call deviceErrorCheck(gpu_istat)
			end do
			call deviceSynchronize(gpu_devices)
			
			! Memory copy H2D
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				!gpu_istat = cudaMemcpy(d_a(:), h_a(GPU,:), n, cudaMemcpyHostToDevice)
				!gpu_istat = cudaMemcpy(d_b(:), h_b(GPU,:), n, cudaMemcpyHostToDevice)
				gpu_istat = cudaMemcpyAsync(d_a(:), h_a(GPU,:), n, cudaMemcpyHostToDevice, g_stream(GPU))
				gpu_istat = cudaMemcpyAsync(d_b(:), h_b(GPU,:), n, cudaMemcpyHostToDevice, g_stream(GPU))
				gpu_istat = cudaStreamSynchronize(g_stream(GPU))
				call deviceErrorCheck(gpu_istat)
			end do
			call deviceSynchronize(gpu_devices)
			
			blockSize = dim3(1024,1,1)
			gridSize = dim3(ceiling(real(n)/real(blockSize%x)) ,1,1)
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				call vecAdd_kernel<<<gridSize, blockSize, g_stream(GPU)>>>(n, d_a, d_b, d_c)
				gpu_istat = cudaStreamSynchronize(g_stream(GPU))
			end do
			call deviceSynchronize(gpu_devices)

			! Memory copy D2H
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				!gpu_istat =  cudaMemcpy(h_c(GPU, :), d_c(:), n, cudaMemcpyDeviceToHost)
				gpu_istat = cudaMemcpyAsync(h_c(GPU, :), d_c(:), n, cudaMemcpyDeviceToHost, g_stream(GPU))
				gpu_istat = cudaStreamSynchronize(g_stream(GPU))
				call deviceErrorCheck(gpu_istat)
			end do
			call deviceSynchronize(gpu_devices)
			
			do GPU = 1, num_gpus
				do i = 1, n
					write(*,*) h_c(GPU, i)
				end do
			end do
			write(*,*) 'finshed'
			
			! CPU & GPU deallocate memory
			deallocate(h_a)
			deallocate(h_b)
			deallocate(h_c)
			do GPU = 1, num_gpus
				deallocate(d_a)
				deallocate(d_b)
				deallocate(d_c)
			end do
		end do
	end do
	
	write(*,*) 'stream Destroy'
	!Gpu stream Destroy
	!do GPU = 1, num_gpus
		!do i =1, num_streams
			gpu_istat = cudaStreamDestroy(g_stream(GPU))
		!end do
	!end do

end program