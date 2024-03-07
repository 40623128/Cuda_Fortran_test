module dataTransfer
	contains
	!dataTransfer_real8_3Dto1D_type0 is used 5 tier do loop
	!dataTransfer_real8_3Dto1D_type1 is used 1 tier do loop(used mod function)
	!dataTransfer_real8_3Dto1D_type2 is used 1 tier do loop(not used mod function)
	! cost time  type0 ≈ type2 < type 1
	subroutine dataTransfer_real8_3Dto1D_type0(a,b)
		implicit none
		integer :: n_x, n_y, n_z
		integer :: i, j, k
		integer:: element, total_element
		integer, dimension(:) ,allocatable :: ashape
		real(8),intent(in),dimension(:,:,:) :: a
		real(8),intent(out),dimension(:) :: b
		
		!allocate(ashape, mold=shape(a))
		ashape = shape(a)
		n_x = ashape(1)
		n_y = ashape(2)
		n_z = ashape(3)
		total_element = n_x*n_y*n_z
		
		do i = 1, n_x
			do j = 1, n_y
				do k = 1, n_z
					element = (i-1)*n_y*n_z &
							+(j-1)*n_z &
							+k
					b(element) = a(i,j,k)
				end do
			end do
		end do
	end subroutine
	
	subroutine dataTransfer_real8_3Dto1D_type1(a,b)
		implicit none
		integer :: n_x, n_y, n_z
		integer :: i, j, k
		integer:: element, total_element
		integer, dimension(:) ,allocatable :: ashape
		real(8),intent(in),dimension(:,:,:) :: a
		real(8),intent(out),dimension(:) :: b
		
		!allocate(ashape, mold=shape(a))
		ashape = shape(a)
		n_x = ashape(1)
		n_y = ashape(2)
		n_z = ashape(3)
		total_element = n_x*n_y*n_z
		
		do element = 1, total_element
			i = mod((element-1)/(n_z*n_y), n_x) + 1
			j = mod((element-1)/n_z, n_y) + 1
			k = mod(element-1, n_z) + 1
			b(element) = a(i,j,k)
		end do
	end subroutine
	
	subroutine dataTransfer_real8_3Dto1D_type2(a,b)
		implicit none
		integer :: n_x, n_y, n_z
		integer :: i, j, k
		integer:: element, total_element
		integer, dimension(:) ,allocatable :: ashape
		real(8),intent(in),dimension(:,:,:) :: a
		real(8),intent(out),dimension(:) :: b
		
		!allocate(ashape, mold=shape(a))
		ashape = shape(a)
		n_x = ashape(1)
		n_y = ashape(2)
		n_z = ashape(3)
		total_element = n_x*n_y*n_z
		
		do element = 1, total_element
			i = (element-1)/(n_y*n_z) + 1
			j = ((element-1)-(i-1)*n_y*n_z)/n_z+1
			k = (element-1)-(i-1)*n_y*n_z-(j-1)*n_z+1
			b(element) = a(i,j,k)
		end do
	end subroutine
end module dataTransfer