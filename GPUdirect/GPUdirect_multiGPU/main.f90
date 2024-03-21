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
	real(8) :: T1, T2, pass_t ,total_time
	! GPU
	integer :: GPU, num_gpus, istat, local_rank
	! CPU
	integer :: rank_gpu,rank, ierr, num_procs, local_comm 
	
	real(8), dimension(:,:), allocatable :: h_a
	real(8), dimension(:,:), allocatable :: h_a_send
	real(8), dimension(:,:), allocatable :: h_a_recv
	
	real(8), device, dimension(:), allocatable  :: d_a
	real(8), device, dimension(:), allocatable  :: d_a_send
	real(8), device, dimension(:), allocatable  :: d_a_recv
	


	
	!MPI init
	call MPI_INIT(ierr)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD, rank ,ierr)
	print*, 'rank/num_procs: ', rank, '/', num_procs
	call gpu_init(num_gpus)
	n = 0
	ntimes = 1

	allocate(gpu_devices(num_gpus))
	do GPU = 1, num_gpus
		gpu_devices(GPU) = GPU - 1
	end do
	
	if (rank .EQ. 1) then
		print *, '================= Start N0D0 -> N1D0 ================='
	end if
	
	do j = 26, 26
		n = 2**j
		total_time = 0
		do k = 1, ntimes
			allocate(h_a(num_gpus,n))
			allocate(h_a_send(num_gpus,n))
			allocate(h_a_recv(num_gpus,n))
			if (rank .EQ. 0) then
				do GPU = 1, num_gpus
					do i = 1,n
						h_a(GPU,i) = GPU
					end do
				end do
			end if
			
			call cpu_time(T1)
			do GPU = 1, num_gpus
				istat=cudaSetDevice(gpu_devices(GPU))
				allocate(d_a(n))
				allocate(d_a_send(n))
				allocate(d_a_recv(n))
				
				if (rank .EQ. 0) then
					istat = cudaMemcpy(d_a_send, h_a(GPU,1), n, cudaMemcpyHostToDevice)
				end if
				
				if (rank .EQ. 0) then
					call MPI_SEND(d_a_send, n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, ierr);
				else if (rank .EQ. 1) then
					call MPI_RECV(d_a_recv, n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, ierr);
				end if
				call MPI_BARRIER(MPI_COMM_WORLD, ierr)
				ierr = cudaDeviceSynchronize()
				if (rank .EQ. 1) then
					istat = cudaMemcpy(h_a(GPU, 1), d_a_recv, n, cudaMemcpyDeviceToHost)
				end if
				if (rank .EQ. 1) then
					print*, 'h_a',h_a(GPU,1)
				end if
			end do
			call cpu_time(T2)
			
			
			
			deallocate(h_a)
			deallocate(h_a_send)
			deallocate(h_a_recv)
			do GPU = 1, num_gpus
				istat=cudaSetDevice(gpu_devices(GPU))
				deallocate(d_a)
				deallocate(d_a_send)
				deallocate(d_a_recv)
			end do

			total_time =total_time + (T2-T1)
		end do
		
		
		if (rank .EQ. 1) then
			if (j .EQ. 1) then
				print *,'pass time(s)','data size(bytes)','Bandwidth(Mb/s)'
			end if
			print *, total_time/ntimes, 2**j, (2**j/(total_time/ntimes)/2**20/8)
		end if
	end do
	
	if (rank .EQ. 1) then
		print *, '================= End N0D0 -> N1D0 ================='
	end if
	
	
	!if (rank .EQ. 1) then
	!	print *, '================= Start N0D0 -> N0 -> N1-> N1D0 ================='
	!end if
	!
	!do j = 28, 28
	!	n = 2**j
	!	total_time = 0
	!	do k = 1, ntimes
	!		allocate(h_a(n))
	!		allocate(h_a_send(n))
	!		allocate(h_a_recv(n))
	!	
	!		allocate(d_a(n))
	!		allocate(d_a_send(n))
	!		allocate(d_a_recv(n))
	!
	!		if (rank .EQ. 0) then
	!			do i = 1,n
	!				h_a(i) = rank*1.d0+1.d0
	!			end do
	!			d_a_send = h_a
	!		end if
	!		ierr = cudaDeviceSynchronize()
	!		call MPI_BARRIER(MPI_COMM_WORLD, ierr)
	!		
	!		call cpu_time(T1)
	!		
	!		
	!		
	!		if (rank .EQ. 0) then
	!			h_a_send = d_a_send
	!			call MPI_SEND(h_a_send, n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, ierr);
	!		else if (rank .EQ. 1) then
	!			call MPI_RECV(h_a_recv, n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, ierr);
	!			d_a = h_a_recv
	!			ierr = cudaDeviceSynchronize()
	!			!print *,h_a
	!		end if
	!		call MPI_BARRIER(MPI_COMM_WORLD, ierr)
	!		call cpu_time(T2)
	!		
	!		deallocate(h_a)
	!		deallocate(h_a_send)
	!		deallocate(h_a_recv)
	!		deallocate(d_a)
	!		deallocate(d_a_send)
	!		deallocate(d_a_recv)
	!		total_time =total_time + (T2-T1)
	!	end do
	!	
	!	if (rank .EQ. 1) then
	!		if (j .EQ. 1) then
	!			print *,'pass time(s)','data size(bytes)','Bandwidth(Mb/s)'
	!		end if
	!		print *, total_time/ntimes, 2**j, (2**j/(total_time/ntimes))/2**20*8
	!	end if
	!	
	!end do
	!if (rank .EQ. 1) then
	!	print *, '================= End N0D0 -> N0 -> N1-> N1D0 ================='
	!end if
	
	call MPI_FINALIZE(ierr)

end program