program mpicuda
	use MPI
	use cudafor
	use cudamod
	use gpu_fuc
	implicit none

	integer :: nx,ny,nz
	integer :: n_outside
	integer :: total_n , each_cube_n
	integer :: ncube
	integer :: n, i, j, k, l
	type(dim3) :: blocksize, gridsize
	integer, dimension(:),allocatable :: gpu_devices
	integer :: ntimes
	integer::status(MPI_STATUS_SIZE)
	! Timer
	real(8) :: T1, T2, pass_t , total_time_1, total_time_2
	! GPU
	integer :: GPU, num_gpus, gpu_istat, local_rank
	! CPU
	integer :: rank_gpu,rank, istat, num_procs, local_comm 
	
	real(8), dimension(:,:,:,:), allocatable :: h_cube
	real(8), dimension(:,:,:,:), allocatable :: h_cube2
	
	real(8), device, dimension(:,:,:,:), allocatable :: d_cube
	real(8), device, dimension(:,:,:)  , allocatable :: d_each_cube
	real(8), device, dimension(:,:,:)  , allocatable :: d_each_cube2
	
	!MPI init
	call MPI_INIT(istat)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, istat)
	call MPI_COMM_RANK(MPI_COMM_WORLD, rank ,istat)
	print*, 'rank/num_procs: ', rank, '/', num_procs
	
	!GPU init
	call gpu_init(num_gpus)
	!num_gpus = 8
	print*, 'num_gpus', num_gpus
	!para init
	n = 0
	ncube = 8
	
	n_outside = 4
	nx = 16
	ny = 16
	nz = 16
	total_n = ncube * (nx+2*n_outside) * (ny+2*n_outside) * (nz+2*n_outside)
	each_cube_n = (nx+2*n_outside) * (ny+2*n_outside) * (nz+2*n_outside)
	!test times
	ntimes = 5

	! allocate cube
	allocate(h_cube(ncube, nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
	allocate(h_cube2(ncube, nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
	
	!give every gpu a index
	allocate(gpu_devices(num_gpus))

	do GPU = 1, num_gpus
		gpu_devices(GPU) = GPU - 1
	end do
	
	do l = 1, ncube
		do i = 1, nx+2*n_outside
			do j = 1, ny+2*n_outside
				do k = 1, nz+2*n_outside
					h_cube(l, i, j, k) = 1.d0
				end do
			end do
		end do
	end do
	
	do GPU = 1, num_gpus
		gpu_istat=cudaSetDevice(gpu_devices(GPU))
		write(*,*) 'allocate d_each_cube'
		allocate(d_each_cube(nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
		allocate(d_each_cube2(nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
		write(*,*) 'allocate d_cube'
		allocate(d_cube(ncube, nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
		write(*,*) 'allocated d_cube'
		!allocate(d_cube(ncube, nx+2*n_outside, ny+2*n_outside, nz+2*n_outside))
	end do
	
	
	blockSize = dim3( 8, 16, 32)
	gridSize = dim3(ceiling(real(nx)/real(blockSize%x)), &
					ceiling(real(ny)/real(blockSize%y)), &
					ceiling(real(nz)/real(blockSize%z)))
	
	write(*,*) 'start add'
	do GPU = 1, num_gpus
		gpu_istat=cudaSetDevice(gpu_devices(GPU))
		do l = 1, ncube
			write(*,*) l
			gpu_istat =  cudaMemcpy(d_each_cube, h_cube(l, :, :, :), each_cube_n, cudaMemcpyHostToDevice)
			!d_cube = h_cube(l, :, :, :)
			call add_gpukern<<<gridSize, blockSize>>>(d_each_cube, nx, ny, nz ,n_outside, d_each_cube2)
			gpu_istat =  cudaMemcpy(h_cube2(l, :, :, :), d_each_cube2, each_cube_n, cudaMemcpyHostToDevice)
		end do
	end do
	
	do l = 1, ncube
		do i = 1, nx+2*n_outside
			do j = 1, ny+2*n_outside
				do k = 1, nz+2*n_outside
					if ((i <= 2) .OR. (i >= nx+2*n_outside -1)) then
						if h_cube2(l, :, :, :) .ne. 1 then
							write(*,*) '(i <= 2) .OR. (i >= nx+2*n_outside -1) Error'
					else
						if h_cube2(l, :, :, :) .ne. 6 then
							write(*,*) '(2 < i  < nx+2*n_outside -1) Error'
				end do
			end do
		end do
	end do
	
	call MPI_FINALIZE(istat)

end program