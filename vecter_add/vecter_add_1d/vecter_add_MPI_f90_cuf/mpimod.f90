module mpimod
	use MPI
	contains
	subroutine mpiinit(myrank, num_procs)
        	implicit none

		integer, intent(out) :: myrank
		integer, intent(out) :: num_procs
		integer :: rank, ierr

		call MPI_INIT(ierr)
		call MPI_COMM_SIZE(MPI_COMM_WORLD, num_procs, ierr)
		call MPI_COMM_RANK(MPI_COMM_WORLD, rank ,ierr)
		myrank = rank

		return
	end subroutine mpiinit

	subroutine finalizempi()

		implicit none
		integer :: ierr

		call MPI_FINALIZE(ierr)

		return

	end subroutine finalizempi
end module mpimod