
file(GLOB cameraDetectionSources *.cpp)
file(GLOB cameraDetectionIncludes *.h )

cuda_add_executable(camera_detection ${cameraDetectionSources})
target_link_libraries(camera_detection jetson-inference)
