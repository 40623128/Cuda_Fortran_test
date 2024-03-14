program mpicuda
	use MPI
	implicit none
	
	integer :: rank, ierr, num_procs, n, i, j, k
	integer :: ntimes
	integer::status(MPI_STATUS_SIZE)
	real(8) :: T1, T2, pass_t ,total_time
	
	
	real(8), dimension(:), allocatable :: h_a
	real(8), dimension(:), allocatable :: h_a_send
	real(8), dimension(:), allocatable :: h_a_recv
	
	real(8), device, dimension(:), allocatable  :: d_a
	real(8), device, dimension(:), allocatable  :: d_a_send
	real(8), device, dimension(:), allocatable  :: d_a_recv
	

	n = 2**28
	ntimes = 10
	

	
	call MPI_INIT(ierr)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD, rank ,ierr)
	print*, 'hello world from rank ', rank
	
	if (rank .EQ. 1) then
		print *, '================= Start N0D0 -> N1D0 ================='
	end if
	do j = 1, 28
		n = 2**j
		total_time = 0
		do k = 1, ntimes
			allocate(h_a(n))
			allocate(h_a_send(n))
			allocate(h_a_recv(n))
		
			allocate(d_a(n))
			allocate(d_a_send(n))
			allocate(d_a_recv(n))

			if (rank .EQ. 0) then
				do i = 1,n
					h_a(i) = rank*1.d0+1.d0
				end do
				d_a_send = h_a
			end if
			call MPI_BARRIER(MPI_COMM_WORLD, ierr)
			
			call cpu_time(T1)
			if (rank .EQ. 0) then
				call MPI_SEND(d_a_send, n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, ierr);
			else if (rank .EQ. 1) then
				call MPI_RECV(d_a_recv, n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, ierr);
				!print *,h_a
			end if
			
			h_a = d_a_recv
			call MPI_BARRIER(MPI_COMM_WORLD, ierr)
			call cpu_time(T2)
			
			deallocate(h_a)
			deallocate(h_a_send)
			deallocate(h_a_recv)
			deallocate(d_a)
			deallocate(d_a_send)
			deallocate(d_a_recv)
			total_time =total_time + (T2-T1)
		end do
		
		if (rank .EQ. 1) then
			if (j .EQ. 1) then
				print *,'pass time(s)','data size(bytes)','Bandwidth(Mb/s)'
			end if
			print *, total_time/ntimes, 2**j, (2**j/(total_time/ntimes))/2**20*8
		end if
	end do
	
	if (rank .EQ. 1) then
		print *, '================= End N0D0 -> N1D0 ================='
	end if
	
	
	if (rank .EQ. 1) then
		print *, '================= Start N0D0 -> N0 -> N1-> N1D0 ================='
	end if
	
	do j = 1, 28
		n = 2**j
		total_time = 0
		do k = 1, ntimes
			allocate(h_a(n))
			allocate(h_a_send(n))
			allocate(h_a_recv(n))
		
			allocate(d_a(n))
			allocate(d_a_send(n))
			allocate(d_a_recv(n))

			if (rank .EQ. 0) then
				do i = 1,n
					h_a(i) = rank*1.d0+1.d0
				end do
				d_a_send = h_a
			end if
			
			call MPI_BARRIER(MPI_COMM_WORLD, ierr)
			
			call cpu_time(T1)
			
			
			
			if (rank .EQ. 0) then
				h_a_send = d_a_send
				call MPI_SEND(h_a_send, n, MPI_REAL8, rank +1, 0, MPI_COMM_WORLD, ierr);
			else if (rank .EQ. 1) then
				call MPI_RECV(h_a_recv, n, MPI_REAL8, rank -1, 0, MPI_COMM_WORLD, status, ierr);
				d_a = h_a_recv
				!print *,h_a
			end if
			call MPI_BARRIER(MPI_COMM_WORLD, ierr)
			call cpu_time(T2)
			
			deallocate(h_a)
			deallocate(h_a_send)
			deallocate(h_a_recv)
			deallocate(d_a)
			deallocate(d_a_send)
			deallocate(d_a_recv)
			total_time =total_time + (T2-T1)
		end do
		
		if (rank .EQ. 1) then
			if (j .EQ. 1) then
				print *,'pass time(s)','data size(bytes)','Bandwidth(Mb/s)'
			end if
			print *, total_time/ntimes, 2**j, (2**j/(total_time/ntimes))/2**20*8
		end if
		
	end do
	if (rank .EQ. 1) then
		print *, '================= End N0D0 -> N0 -> N1-> N1D0 ================='
	end if
	
	call MPI_FINALIZE(ierr)

end program