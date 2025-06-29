set(TARGET sd-qt6)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Enable Qt6 MOC processing
set(CMAKE_AUTOMOC ON)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

MESSAGE(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

target_include_directories(
    ${TARGET} 
    PRIVATE ${CMAKE_SOURCE_DIR}/../
)

target_link_libraries(
    ${TARGET} 
    PRIVATE 
    stable-diffusion 
    ${CMAKE_THREAD_LIBS_INIT} 
    Qt6::Core Qt6::Widgets
)

target_compile_features(
    ${TARGET} 
    PUBLIC 
    cxx_std_17
)