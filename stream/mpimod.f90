module mpimod
	use MPI
	contains
	subroutine mpiinit(myrank)
        	implicit none

		integer, intent(out) :: myrank
		integer :: rank, ierr, num_procs

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