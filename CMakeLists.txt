cmake_minimum_required(VERSION 3.0)

set(OpenCV_DIR C:/opencv-4.1.1-mingw/)
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE Debug)

get_filename_component(ProjectId ${CMAKE_CURRENT_LIST_DIR} NAME)

string(REPLACE " " "_" ProjectId ${ProjectId})

project(${ProjectId} C CXX)

set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
 
file(GLOB SOURCES  *.cpp)
 
add_executable(${PROJECT_NAME} ${SOURCES}  )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
