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
	use gputype
	implicit none
	
	real(8) :: T1, T2, pass_t ,total_time
	type(dim3) :: blockSize, gridSize
	type(TGPUplan), dimension(:),allocatable :: myGPUplan
	real(8) :: sum_num
	integer :: i, j, k, l, m
	
	! Size of vectors
	integer :: total_cube, gpu_cube
	integer :: n, n_outside, total_n
	integer :: n_x, n_x_total
	integer :: n_y, n_y_total
	integer :: n_z, n_z_total
	integer :: gpu_n
	integer :: istat, ierr, GPU
	integer :: num_gpus,devnum
	
	real(8),dimension(:,:,:,:),allocatable :: all_mesh_1
	real(8),dimension(:,:,:,:),allocatable :: all_mesh_2
	real(8),dimension(:,:,:,:),allocatable :: all_mesh_3
	
	! Device input vectors
	integer, dimension(:),allocatable :: gpu_devices
	
	type(cudadeviceprop):: prop
	
	! GPU Initialize
	call gpu_init(num_gpus)
	allocate(myGPUplan(num_gpus))
	
	n_x = 16
	n_y = 16
	n_z = 16
	n = n_x * n_y * n_z
	
	n_outside = 4
	n_x_total = n_x + 2*n_outside
	n_y_total = n_y + 2*n_outside
	n_z_total = n_z + 2*n_outside
	
	total_cube = 2400
	gpu_cube = total_cube/num_gpus
	total_n = total_cube * n_x_total * n_y_total * n_z_total
	gpu_n = gpu_cube * n_x_total * n_y_total * n_z_total
	allocate(gpu_devices(num_gpus))
	!allocate(g_stream(num_gpus))
	do GPU = 1, num_gpus
        gpu_devices(GPU) = GPU - 1
    end do
	! Allocate memory for all mesh on host
	allocate(all_mesh_1(total_cube, n_x_total, n_y_total, n_z_total))
	allocate(all_mesh_2(total_cube, n_x_total, n_y_total, n_z_total))
	allocate(all_mesh_3(total_cube, n_x_total, n_y_total, n_z_total))
	
	do GPU = 1, num_gpus
		allocate(myGPUplan(GPU)%gpu_all_mesh_1(gpu_cube, n_x_total, n_y_total, n_z_total))
		allocate(myGPUplan(GPU)%gpu_all_mesh_2(gpu_cube, n_x_total, n_y_total, n_z_total))
	end do
	
	print*, 'allocate host memory completed'
	
	
	!Initialize content of mesh, mesh(x,y,z) = 1
	do l = 1, total_cube
		do i = 1, n_x_total
			do j = 1, n_y_total
				do k = 1, n_z_total
					all_mesh_1(l,i,j,k) = i+j+k
					all_mesh_2(l,i,j,k) = 0.d0
					all_mesh_3(l,i,j,k) = 0.d0
				end do
			end do
		end do
	end do
	print*, 'Initialize content of mesh completed'
	
	do GPU = 1, num_gpus
		do l = 1, gpu_cube
			do i = 1, n_x_total
				do j = 1, n_y_total
					do k = 1, n_z_total
						myGPUplan(GPU)%gpu_all_mesh_1(l, i, j, k) = all_mesh_1((GPU-1)*gpu_cube+l,i,j,k)
					end do
				end do
			end do
		end do
	end do
	
	!check device
	do GPU = 1, num_gpus
		print *, 'gpu_devices = ',gpu_devices(GPU)
		istat=cudaSetDevice(gpu_devices(GPU))
		istat=cudaGetDeviceProperties(prop,gpu_devices(GPU))
		print *, 'device name = ',trim(prop%name)
		istat = cudaStreamCreate(myGPUplan(GPU)%stream1)
	end do
	
	! allocate gpu memory
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		allocate(myGPUplan(GPU)%d_mesh_1(gpu_cube, n_x_total, n_y_total, n_z_total))
		allocate(myGPUplan(GPU)%d_mesh_2(gpu_cube, n_x_total, n_y_total, n_z_total))
		allocate(myGPUplan(GPU)%gpu_mesh_1(n_x_total, n_y_total, n_z_total))
		allocate(myGPUplan(GPU)%gpu_mesh_2(n_x_total, n_y_total, n_z_total))
		if (.not. allocated(myGPUplan(GPU)%d_mesh_1)) then
			print *, "Error allocating memory for d_mesh_1 on GPU", GPU
		end if
		if (.not. allocated(myGPUplan(GPU)%d_mesh_2)) then
			print *, "Error allocating memory for d_mesh_2 on GPU", GPU
		end if
		if (.not. allocated(myGPUplan(GPU)%gpu_mesh_1)) then
			print *, "Error allocating memory for gpu_mesh_1 on GPU", GPU
		end if
		if (.not. allocated(myGPUplan(GPU)%gpu_mesh_2)) then
			print *, "Error allocating memory for gpu_mesh_2 on GPU", GPU
		end if
	end do
	
	!start calculating
	call cpu_time(T1)
	print *, 'HtoD Start'
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaMemcpyAsync(myGPUplan(GPU)%d_mesh_1, myGPUplan(GPU)%gpu_all_mesh_1, gpu_n, cudaMemcpyHostToDevice, myGPUplan(GPU)%stream1)
	end do
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaStreamSynchronize(myGPUplan(GPU)%stream1)
	end do
	print *, 'HtoD Finshed'
	
	print *, 'meshAdd_1Dkernel start'
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		blockSize = dim3(8,8,8)
		gridSize = dim3(ceiling(real(n_x_total)/real(blockSize%x)), &
						ceiling(real(n_y_total)/real(blockSize%y)), &
						ceiling(real(n_z_total)/real(blockSize%z)))
		call meshAdd_3Dkernel<<<gridSize, blockSize, 0, myGPUplan(GPU)%stream1>>>(myGPUplan(GPU)%d_mesh_1, gpu_cube, n_x_total, n_y_total, n_z_total, n_outside, myGPUplan(GPU)%d_mesh_2)
		call meshAdd_3Dkernel<<<gridSize, blockSize, 0, myGPUplan(GPU)%stream1>>>(myGPUplan(GPU)%d_mesh_2, gpu_cube, n_x_total, n_y_total, n_z_total, n_outside, myGPUplan(GPU)%d_mesh_1)
		call meshAdd_3Dkernel<<<gridSize, blockSize, 0, myGPUplan(GPU)%stream1>>>(myGPUplan(GPU)%d_mesh_1, gpu_cube, n_x_total, n_y_total, n_z_total, n_outside, myGPUplan(GPU)%d_mesh_2)
		call meshAdd_3Dkernel<<<gridSize, blockSize, 0, myGPUplan(GPU)%stream1>>>(myGPUplan(GPU)%d_mesh_2, gpu_cube, n_x_total, n_y_total, n_z_total, n_outside, myGPUplan(GPU)%d_mesh_1)
	end do
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaStreamSynchronize(myGPUplan(GPU)%stream1)
	end do
	print *, 'meshAdd_1Dkernel Finshed'
	
	print *, 'DtoH Start'
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		!istat = cudaMemcpy(myGPUplan(GPU)%gpu_all_mesh_2, myGPUplan(GPU)%d_mesh_2, gpu_n, cudaMemcpyDeviceToHost)
		istat = cudaMemcpy(myGPUplan(GPU)%gpu_all_mesh_2, myGPUplan(GPU)%d_mesh_1, gpu_n, cudaMemcpyDeviceToHost)

	end do
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaStreamSynchronize(myGPUplan(GPU)%stream1)
	end do
	print *, 'DtoH Finshed'
	call cpu_time(T2)
	
	! gpu list to another list
	do GPU = 1, num_gpus
		do l = 1, gpu_cube
			do i = 1, n_x_total
				do j = 1, n_x_total
					do k = 1, n_x_total
						all_mesh_2((GPU-1)*gpu_cube+l,i,j,k) = myGPUplan(GPU)%gpu_all_mesh_2(l, i, j, k)
					end do
				end do
			end do
		end do
	end do
	do GPU = 1, num_gpus
		istat=cudaSetDevice(gpu_devices(GPU))
		istat = cudaDeviceSynchronize()
	end do
	
	print*,'CPU Calculating '
	call cpu_time(T1)
	do m = 1, n_outside
		do l = 1, total_cube
			do i = 1+m, n_x_total-m
				do j = 1+m, n_x_total-m
					do k = 1+m, n_x_total-m
						all_mesh_3(l,i,j,k) = all_mesh_1(l, i+1, j, k) + all_mesh_1(l, i, j+1, k) + all_mesh_1(l, i, j, k+1) &
											+ all_mesh_1(l, i-1, j, k) + all_mesh_1(l, i, j-1, k) + all_mesh_1(l, i, j, k-1)
					end do
				end do
			end do
		end do
		all_mesh_1 = all_mesh_3
	end do
	call cpu_time(T2)
	print *, "Elapsed time:", T2-T1, "seconds"
	print*,'CPU Calculate Completed'
	
	
	!do l = 1, total_cube
	!	do i = 1+n_outside, n_x_total-n_outside
	!		do j = 1+n_outside, n_x_total-n_outside
	!			do k = 1+n_outside, n_x_total-n_outside
	!				if (all_mesh_2(l, i, j, k) /= all_mesh_3(l, i, j, k)) then
	!					print*,'l =',l,'i =',i, 'j =',j,'k =',k
	!				end if
	!			end do
	!		end do
	!	end do
	!end do
	print*,all_mesh_2(5, 8, 8, 8)
	print*,all_mesh_3(5, 8, 8, 8)
 
 
    print *, 'end '
 
    ! Release device memory
    deallocate(all_mesh_1)
    deallocate(all_mesh_2)
    deallocate(all_mesh_3)
 
    ! Release host memory
	do GPU = 1, num_gpus
		deallocate(myGPUplan(GPU)%d_mesh_1)
		deallocate(myGPUplan(GPU)%d_mesh_2)
	end do
 
end program main