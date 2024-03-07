module dataTransfer
	contains
	implicit none
	!a is 3d array
	!b is 1d array
	!this function make
	!a(n_x,n_y,n_z) = b(n_x*n_y*n_z)
	!a(1,1,1) = b(1)
	!a(1,1,2) = b(2)
	!a(1,2,1) = b(n_x+1)
	subroutine dataTransfer_real8_3Dto1D(a,b)
	
	integer :: n_x, n_y, n_z
	integer:: total_element
	real(8),dimension(:,:,:) :: a
	real(8),dimension(:) :: b
	
	n_x = sizeof(a(:,:,:))
	n_y = sizeof(a(1,:,:))
	n_z = sizeof(a(1,1,:))
	total_element = n_x*n_y*n_z
	
	do element = 1, total_element
		i = mod((element-1)/(n_z*n_y), n_x) + 1
		j = mod((element-1)/n_z, n_y) + 1
		k = mod(element-1, n_z) + 1
		b(element) = a(i,j,k)
	end do
 
end program main