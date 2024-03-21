program GPUdirect
	!source: https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/
	!To enable GPUDirect the following steps must be taken before invoking aprun:
	!export MPICH_RDMA_ENABLED_CUDA=1
	
    use cudafor
    use MPI
    implicit none
    integer :: direct
    character(len=255) :: env_var
	integer :: size, ierror
    integer :: rank
	integer :: i, j
	integer:: data_size
	integer:: ntimes
	real(8) :: T1, T2, time, ave_time
    integer,dimension(:),allocatable :: h_buff
    integer,dimension(:),allocatable,device :: d_rank_send
    integer,dimension(:),allocatable,device :: d_buff

    call getenv("MPICH_RDMA_ENABLED_CUDA", env_var)
    read( env_var, '(i10)' ) direct
    if (direct .NE. 1) then
      print *, 'MPICH_RDMA_ENABLED_CUDA not enabled!'
      call exit(1)
    endif
 
    call MPI_INIT(ierror)
 
    ! Get MPI rank and size
    call MPI_COMM_RANK (MPI_COMM_WORLD, rank, ierror)
    call MPI_COMM_SIZE (MPI_COMM_WORLD, size, ierror)
	

	ntimes = 10
	time = 0
	data_size = 100
	allocate(h_buff(data_size))
	allocate(d_buff(data_size))
	allocate(d_rank_send(data_size))
	

		T1 = 0
		T2 = 0
		! Initialize host and device buffers
		do i = 1, data_size
			h_buff(i) = 1
		end do
		
		! Implicity copy rank to device
		d_rank_send = h_buff
	
		! Preform allgather using device buffers
		do j = 1 , ntimes
		call MPI_ALLGATHER(d_rank_send, data_size, MPI_INTEGER, d_buff, data_size*rank, MPI_INTEGER, MPI_COMM_WORLD, ierror);
		end do
		time = time + (T2 - T1)
		print *, 'time: ',time

		! Check that buffer is correct
		h_buff = d_buff(1:data_size*rank)
		
		do i=1,data_size*rank
			if (h_buff(i) .NE. 1) then
				print *, 'Alltoall Failed!'
				call exit(1)
			endif
		enddo
		if (rank .EQ. 0) then
			print *, 'Success!'
		endif
	ave_time = time/ntimes
	print *, 'ave time: ',ave_time
    ! Clean up
	deallocate(h_buff)
	deallocate(d_buff)
	deallocate(d_rank_send)
	
    call MPI_FINALIZE(ierror)
 
end program GPUdirect