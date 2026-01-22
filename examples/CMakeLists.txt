include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)

# Only build Qt6 example if Qt6 is available
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
add_subdirectory(qt6)

add_subdirectory(server)
