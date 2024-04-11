!!!!!!!!!!!!!!!!!!!!!!
!**********************
!** 1 ** 1 ** 1 ** 1 **
!**********************
!** 1 ** 1 ** 1 ** 1 **
!**********************
!** 1 ** 1 ** 1 ** 1 **
!**********************
!** 1 ** 1 ** 1 ** 1 **
!**********************

!**********************
!** 2 ** 3 ** 3 ** 2 **
!**********************
!** 3 ** 4 ** 4 ** 3 **
!**********************
!** 3 ** 4 ** 4 ** 3 **
!**********************
!** 2 ** 3 ** 3 ** 2 **
!**********************



program main
	use cudafor
	use kernel
	use cudamod
	implicit none


	type(dim3) :: blockSize, gridSize
	real(8) :: sum_num
	integer :: i, j, k
	
	! Size of vectors
	integer :: n, n_outside, total_n
	integer :: n_x, n_x_total
	integer :: n_y, n_y_total
	integer :: n_z, n_z_total
	integer :: gpu_n
	integer :: istat, ierr, GPU
	integer :: num_gpus,devnum
	! Host input vectors
	real(8),dimension(:,:,:),allocatable :: all_mesh_1
	real(8),dimension(:,:,:),allocatable :: all_mesh_2
	real(8),dimension(:,:,:),allocatable :: all_mesh_3
	
	! Device input vectors
	integer, dimension(:),allocatable :: gpu_devices
	real(8), device, dimension(:,:,:), allocatable :: d_mesh_1
	real(8), device, dimension(:,:,:), allocatable :: d_mesh_2
	!Host output vector
	real(8), device, dimension(:), allocatable :: d_c
	
	!integer(kind=cuda_stream_kind), dimension(:), allocatable ::g_stream
	type(cudadeviceprop):: prop
	!type(cudaEvent) :: event
	
	! GPU Initialize
	call gpu_init(num_gpus)
	num_gpus = 1
	
	n_x = 126
	n_y = 126
	n_z = 126
	n = n_x * n_y * n_z
	
	n_outside = 2
	n_x_total = n_x + 2*n_outside
	n_y_total = n_y + 2*n_outside
	n_z_total = n_z + 2*n_outside
	
	total_n = n_x_total * n_y_total * n_z_total
	
	allocate(gpu_devices(num_gpus))
	!allocate(g_stream(num_gpus))
	do GPU = 1, num_gpus
        gpu_devices(GPU) = GPU - 1
    end do
	! Allocate memory for all mesh on host
	allocate(all_mesh_1(n_x_total, n_y_total, n_z_total))
	allocate(all_mesh_2(n_x_total, n_y_total, n_z_total))
	allocate(all_mesh_3(n_x_total, n_y_total, n_z_total))
	
	print*, 'allocate host memory completed'
	
	
	!Initialize content of mesh, mesh(x,y,z) = 1
	do i = 1, n_x_total
		do j = 1, n_x_total
			do k = 1, n_x_total
				all_mesh_1(i,j,k) = 1.d0
				all_mesh_2(i,j,k) = 0.d0
				all_mesh_3(i,j,k) = 0.d0
			end do
		end do
    end do
	print*, 'Initialize content of mesh completed'
	
	!check device
	do GPU = 1, num_gpus
		print *, 'gpu_devices = ',gpu_devices(GPU)
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaGetDeviceProperties(prop,gpu_devices(GPU))
		print *, 'device name = ',trim(prop%name)
	end do
	
	!start calculating
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		allocate(d_mesh_1(n_x_total, n_y_total, n_z_total))
		allocate(d_mesh_2(n_x_total, n_y_total, n_z_total))
		istat=cudaMemcpy(d_mesh_1, all_mesh_1, total_n, cudaMemcpyHostToDevice)
		istat=cudaMemcpy(d_mesh_2, all_mesh_2, total_n, cudaMemcpyHostToDevice)
		blockSize = dim3(8,8,8)
		gridSize = dim3(ceiling(real(n_x_total)/real(blockSize%x)) ,ceiling(real(n_y_total)/real(blockSize%y)),ceiling(real(n_z_total)/real(blockSize%z)))
		!istat = cudaEventSynchronize(event)
		call meshAdd_3Dkernel<<<gridSize, blockSize>>>(d_mesh_1, n_x_total, n_y_total, n_z_total, n_outside, d_mesh_2)
		!call meshAdd2Tier_3Dkernel<<<gridSize, blockSize>>>(d_mesh_1, n_x_total, n_y_total, n_z_total, n_outside, d_mesh_2)
	end do
	do GPU = 1, num_gpus
		istat = cudaSetDevice(gpu_devices(GPU))
		istat = cudaDeviceSynchronize()
	end do
	print*,'calculate completed'
	
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaMemcpy(all_mesh_2, d_mesh_2, total_n, cudaMemcpyDeviceToHost)
	end do
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaDeviceSynchronize()
	end do
	print*,'copy data of device to host completed'
	
	print*,'CPU Calculating '
	do i = 2, n_x_total-1
		do j = 2, n_x_total-1
			do k = 2, n_x_total-1
				all_mesh_3(i,j,k) = all_mesh_1(i+1, j, k) + all_mesh_1(i, j+1, k) + all_mesh_1(i+1, j+1, k)&
									+ all_mesh_1(i-1, j, k) + all_mesh_1(i, j-1, k) + all_mesh_1(i, j, k-1)
			end do
		end do
    end do
	print*,'CPU Calculate Completed'
	
	
	do i = 3, n_x_total-2
		do j = 3, n_x_total-2
			do k = 3, n_x_total-2
				if (all_mesh_2(i, j, k) /= all_mesh_3(i, j, k)) then
					print*,'i =',i, 'j =',j,'k =',k
				end if
			end do
		end do
    end do
	print*,all_mesh_2(20,20,20)
	print*,all_mesh_3(20,20,20)
 
 
    print *, 'end '
 
    ! Release device memory
    deallocate(all_mesh_1)
    deallocate(all_mesh_2)
    deallocate(all_mesh_3)
 
    ! Release host memory
    deallocate(d_mesh_1)
    deallocate(d_mesh_2)
 
end program main