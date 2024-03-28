module gpu_fuc
	use cudafor
	contains

	subroutine deviceErrorCheck(gpu_istat, error_msg)
		implicit none
		character(len=*), intent(in), optional :: error_msg
		integer, intent(in) :: gpu_istat
		if (gpu_istat /= cudaSuccess) then
			write(*,*) 'Error:',error_msg
			write(*,*) cudaGetErrorString(gpu_istat)
		end if
		return
	end subroutine deviceErrorCheck
	
	subroutine deviceSynchronize(gpu_devices)
		implicit none
		integer :: num_gpus
		integer, intent(in) :: gpu_devices(:)
		integer :: GPU
		integer :: gpu_istat
		character(len=100) :: error_msg
		num_gpus = size(gpu_devices)
		
		do GPU = 1, num_gpus
			gpu_istat=cudaSetDevice(gpu_devices(GPU))
			gpu_istat=cudaDeviceSynchronize()
			if (gpu_istat /= cudaSuccess) then
				error_msg = 'Call to cudaDeviceSynchronize() failed'
			else
				error_msg = 'No error'
			end if
			call deviceErrorCheck(gpu_istat)
		end do
		return
	end subroutine deviceSynchronize
end module gpu_fuc