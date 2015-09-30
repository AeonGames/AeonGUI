if(WIN32)
	subdirs(
	# Temporarly removing WindowsOpenGL in favor of ProofOfConcept
	#WindowsOpenGL 
	ProofOfConcept)
endif()
if(UNIX)
	subdirs(LinuxOpenGL)
endif()
