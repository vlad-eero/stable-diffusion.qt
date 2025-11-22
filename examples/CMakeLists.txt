include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)

# Only build Qt6 example if Qt6 is available
find_package(Qt6 QUIET COMPONENTS Core Widgets)
if(Qt6_FOUND)
    add_subdirectory(qt6)
else()
    message(STATUS "Qt6 not found, skipping Qt6 example")
endif()