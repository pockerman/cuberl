ADD_SUBDIRECTORY(intro_example_1)
ADD_SUBDIRECTORY(intro_example_2)
ADD_SUBDIRECTORY(intro_example_3)
ADD_SUBDIRECTORY(intro_example_4)
ADD_SUBDIRECTORY(intro_example_6)
ADD_SUBDIRECTORY(intro_example_7)

# we need CUDA for this to work
IF(USE_CUDA)
	ADD_SUBDIRECTORY(intro_example_8)
ENDIF()
