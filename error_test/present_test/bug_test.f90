module bug_test
	contains
	subroutine present_test(a)
        implicit none
		integer,intent(in), optional ::a
		print*, 'start test present'
		if (present(a)) then
			print*, 'a had input!'
			if (a .le. 5) then
				print*, 'a <5'
			end if
		else
			print*, 'a not had input!'
		end if
		print*, 'start test present + <'
		if (present(a) .and. a .le. 5) then
			print*, 'start & a<5'
		end if
		
		return
	end subroutine present_test
end module bug_test
