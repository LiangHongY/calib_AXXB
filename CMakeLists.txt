cmake_minimum_required(VERSION 3.10)
project(calib_AXXB)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_custom_target(CreateBin ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/bin)
add_custom_target(CreateLib ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

# glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDEDIR}
    ${CERES_INCLUDE_DIRS}
  )

add_executable(${PROJECT_NAME}_lidar_rtk 
                app/calib_lidar_rtk.cpp
                src/hand_eye_calibrator.cpp
            )
target_link_libraries(${PROJECT_NAME}_lidar_rtk 
                            yaml-cpp
                            glog
                            ${CERES_LIBRARIES} 
                        )
set_target_properties(${PROJECT_NAME}_lidar_rtk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
