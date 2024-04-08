program mpicuda
	use MPI
	use cudafor
	use cudamod
	implicit none

	integer :: n, i, j, k
	integer, dimension(:),allocatable :: gpu_devices
	integer :: ntimes
	integer::status(MPI_STATUS_SIZE)
	! Timer
	real(8) :: T1, T2, pass_t , total_time_1, total_time_2
	! GPU
	integer :: GPU, num_gpus, gpu_istat, local_rank
	! CPU
	integer :: rank_gpu,rank, istat, num_procs, local_comm 
	
	real(8), dimension(:,:), allocatable :: h_a
	real(8), dimension(:,:), allocatable :: h_a_send
	real(8), dimension(:,:), allocatable :: h_a_recv
	
	real(8), device, dimension(:), allocatable  :: d_a
	real(8), device, dimension(:), allocatable  :: d_a_send
	real(8), device, dimension(:), allocatable  :: d_a_recv
	
	!MPI init
	call MPI_INIT(istat)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, istat)
	call MPI_COMM_RANK(MPI_COMM_WORLD, rank ,istat)
	print*, 'rank/num_procs: ', rank, '/', num_procs
	
	!GPU init
	call gpu_init(num_gpus)
	!num_gpus = 1
	print*, 'num_gpus', num_gpus
	!para init
	n = 0
	
	!test times
	ntimes = 1

	!give every gpu a index
	allocate(gpu_devices(num_gpus))

	do GPU = 1, num_gpus
		gpu_devices(GPU) = GPU - 1
		!gpu_devices(GPU) = 7
	end do
	
	!if (rank .EQ. 1) then
	!	print *, '================= Start N0D0 -> N1D0 ================='
	!end if
	
	! Start testing
	! j is n's size (if j = 20 then n = 2^20 ......) 
	! n is data size
	do j = 20, 20
		n = 2**j
		total_time_1 = 0
		total_time_2 = 0
		
		do k = 1, ntimes
			! CPU memory Allocate
			! h_a_send for mpi send
			! h_a_recv for mpi recv
			allocate(h_a(num_gpus,n))
			allocate(h_a_send(num_gpus,n))
			allocate(h_a_recv(num_gpus,n))
			
			! Generation problen
			! gpu 1 = 1, gpu 2 =2,...
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					do i = 1,n
						h_a(GPU,i) = GPU
					end do
				end do
			end if
			
			!GPU memory Allocate
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				allocate(d_a(n))
				allocate(d_a_send(n))
				allocate(d_a_recv(n))
			end do
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				gpu_istat=cudaDeviceSynchronize()
			end do
			
			! Memory copy H2D
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat = cudaMemcpy(d_a_send, h_a(GPU,1), n, cudaMemcpyHostToDevice)
				end do
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat=cudaDeviceSynchronize()
				end do
			end if	
			
			call MPI_BARRIER(MPI_COMM_WORLD, istat)
			! Timer records time 1.
			call cpu_time(T1)
			! MPI Send & Recv (GPUDirect RDMA)(GPU to GPU)
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					call MPI_SEND(d_a_send, n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, istat)
				end do
			else if (rank .EQ. 1) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					call MPI_RECV(d_a_recv, n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, istat)
				end do
			end if
			! Timer records time 2.
			call cpu_time(T2)
			
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				gpu_istat=cudaDeviceSynchronize()
			end do
			call MPI_BARRIER(MPI_COMM_WORLD, istat)
			

			
			! Memory copy D2H
			if (rank .EQ. 1) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat = cudaMemcpy(h_a(GPU, 1), d_a_recv, n, cudaMemcpyDeviceToHost)
				end do
			end if
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				gpu_istat=cudaDeviceSynchronize()
			end do
			
			!! print the anser to check the result
			!if (rank .EQ. 1) then
			!	do GPU = 1, num_gpus
			!		print*, 'h_a',h_a(GPU,2)
			!	end do
			!end if
			!call MPI_BARRIER(MPI_COMM_WORLD, istat)
			
			! CPU & GPU deallocate memory
			deallocate(h_a)
			deallocate(h_a_send)
			deallocate(h_a_recv)
			do GPU = 1, num_gpus
				istat=cudaSetDevice(gpu_devices(GPU))
				deallocate(d_a)
				deallocate(d_a_send)
				deallocate(d_a_recv)
			end do
			
			! Calculate the elapsed time
			total_time_1 =total_time_1 + (T2-T1)
		end do
		
		if (rank .EQ. 1) then
			print *,'elapsed time(s)','      data size(bytes)','      Bandwidth(Mb/s)'
			print *, '======================= N0D0 -> N1D0 ======================='
			print *, total_time_1/ntimes, 2**j*num_gpus*8, (2**j*num_gpus*8/(total_time_1/ntimes)/2**20)
		end if


		do k = 1, ntimes
			! CPU memory Allocate
			! h_a_send for mpi send
			! h_a_recv for mpi recv
			allocate(h_a(num_gpus,n))
			allocate(h_a_send(num_gpus,n))
			allocate(h_a_recv(num_gpus,n))
			
			! Generation problen
			! gpu 1 = 1, gpu 2 =2,...
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					do i = 1,n
						h_a(GPU,i) = GPU
					end do
				end do
			end if
			
			!GPU memory Allocate
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				allocate(d_a(n))
				allocate(d_a_send(n))
				allocate(d_a_recv(n))
			end do
			do GPU = 1, num_gpus
				gpu_istat=cudaSetDevice(gpu_devices(GPU))
				gpu_istat=cudaDeviceSynchronize()
			end do
			
			! Memory copy H2D
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat = cudaMemcpy(d_a_send, h_a(GPU,1), n, cudaMemcpyHostToDevice)
				end do
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat=cudaDeviceSynchronize()
				end do
			end if	
			
			call MPI_BARRIER(MPI_COMM_WORLD, istat)
			! Timer records time 1.
			call cpu_time(T1)
			
			! Memory copy D2H
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat = cudaMemcpy(h_a_send(GPU,1), d_a_send, n, cudaMemcpyDeviceToHost)
				end do
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat=cudaDeviceSynchronize()
				end do
			end if	
			
			! MPI send & recv
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					call MPI_SEND(h_a_send(GPU,1), n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, istat)
				end do
			else if (rank .EQ. 1) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					call MPI_RECV(h_a_recv(GPU,1), n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, istat)
				end do
			end if
			
			! Memory copy H2D
			if (rank .EQ. 1) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat = cudaMemcpy(d_a_send, h_a_recv(GPU,1), n, cudaMemcpyHostToDevice)
				end do

			end if	
			
			call cpu_time(T2)
			if (rank .EQ. 1) then
				do GPU = 1, num_gpus
					gpu_istat=cudaSetDevice(gpu_devices(GPU))
					gpu_istat=cudaDeviceSynchronize()
				end do
			end if	
			call MPI_BARRIER(MPI_COMM_WORLD, istat)
			
			deallocate(h_a)
			deallocate(h_a_send)
			deallocate(h_a_recv)
			deallocate(d_a)
			deallocate(d_a_send)
			deallocate(d_a_recv)
			total_time_2 =total_time_2 + (T2-T1)
		end do
		
		if (rank .EQ. 1) then
			print *, '================= N0D0 -> N0 -> N1-> N1D0 ================='
			print *, total_time_2/ntimes, 2**j*num_gpus*8, (2**j*num_gpus*8/(total_time_2/ntimes)/2**20)
			print *, '                      '
		end if
		
		! print the result of Bandwidth test

		
	end do
	!if (rank .EQ. 1) then
	!	print *, '================= End N0D0 -> N0 -> N1-> N1D0 ================='
	!end if
	
	call MPI_FINALIZE(istat)

end program