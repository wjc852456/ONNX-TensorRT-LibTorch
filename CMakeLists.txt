cmake_minimum_required(VERSION 3.0)
project(jetson-inference)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/build.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/build.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/build.cmake)
  endif()
endif()

# submodule warning
message(" ")
message("Note:  this project uses git submodules in the source tree.")
message("       if you haven't already, run the following command from")
message("       the project's root directory:")
message(" ")
message("           git submodule update --init") 
message("\n")

if( NOT EXISTS "${PROJECT_SOURCE_DIR}/utils/.git" )
	message("Note:  required git submodules have not been detected.")
	message("       first, please run the following command from the")
	message("       the project's root directory to clone them:")
	message(" ")
	message("          git submodule update --init")
	message(" ")
	message(FATAL_ERROR "missing required git submodules, see instructions above")
endif()


# setup build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings -Wno-deprecated-declarations")	# -std=c++14 
set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_INTERACTIVE "YES" CACHE BOOL "If NO, will download/install the default DNN models without prompting the user, and skip installation of PyTorch.")


# copy configuration tools to build dir
file(COPY "tools/download-models.sh" DESTINATION ${PROJECT_BINARY_DIR})
file(COPY "tools/download-models.rc" DESTINATION ${PROJECT_BINARY_DIR})


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("-- Launching pre-build dependency installer script...")
	message("-- Build interactive:  ${BUILD_INTERACTIVE}")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_INTERACTIVE}
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("-- Finished installing dependencies")
endif()

# setup libtorch
find_package(Torch REQUIRED
			 PATHS ${TORCH_LIB_PATH})

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda" )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(
	CUDA_NVCC_FLAGS
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72 and more")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		-gencode arch=compute_72,code=sm_72
		${CUDA_ARCH_FLAGS}
	)
endif()

message("-- cmake build type: ${CMAKE_BUILD_TYPE}")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O0)
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
endif()

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build C/C++ library
include_directories(${TRT_INCLUDE_DIR})
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-inference ${PROJECT_INCLUDE_DIR}/jetson-utils)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/glib-2.0/include/)

file(GLOB inferenceSources c/*.cpp c/*.cu calibration/*.cpp)
file(GLOB inferenceIncludes c/*.h c/*.cuh calibration/*.h)

find_package(OpenCV)
if( NOT OpenCV_FOUND )
	message("-- didn't find OpenCV on system, disabling OpenCV")
else()
	message("-- OpenCV version:  " ${OpenCV_VERSION})

	if( ${OpenCV_VERSION_MAJOR} LESS 3 )
		message("-- OpenCV version less than 3.0, disabling OpenCV")
	else()
		message("-- OpenCV version >= 3.0.0, enabling OpenCV")
		set(HAS_OPENCV 1)
		add_definitions(-DHAS_OPENCV)
	endif()	
endif()

cuda_add_library(jetson-inference SHARED ${inferenceSources} ${pluginSources})


# transfer all headers to the include directory
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/jetson-inference)

foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR}/jetson-inference COPYONLY)
endforeach()

# create symbolic link for network and image data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/images" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/images" )


# build subdirectories
add_subdirectory(examples)
#add_subdirectory(tools)
add_subdirectory(utils)


# set linker options
target_link_libraries(jetson-inference jetson-utils nvinfer nvinfer_plugin nvcaffe_parser)
target_link_libraries(jetson-inference ${TORCH_LIBRARIES})

if(CUDA_VERSION_MAJOR GREATER 9)
	target_link_libraries(jetson-inference nvonnxparser)

    if(HAS_OPENCV) 
        message("-- Linking jetson-inference with OpenCV " ${OpenCV_VERSION})
        target_link_libraries(jetson-inference opencv_core opencv_calib3d ${OpenCV_LIBS})
    endif()
endif()


# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/jetson-inference)
endforeach()

# install symlink to networks and images
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/networks ${CMAKE_INSTALL_PREFIX}/bin/networks )" )
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/images ${CMAKE_INSTALL_PREFIX}/bin/images )" )

# install the shared library
install(TARGETS jetson-inference DESTINATION lib EXPORT jetson-inferenceConfig)

# install the cmake project, for importing
install(EXPORT jetson-inferenceConfig DESTINATION share/jetson-inference/cmake)

# run ldconfig after installing
install(CODE "execute_process( COMMAND ldconfig )")

