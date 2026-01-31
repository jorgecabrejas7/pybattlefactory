cmake_minimum_required(VERSION 3.16)
project(pokemon_battle_simulator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core simulator library
add_library(battle_sim STATIC
    src/types.cpp
    src/battle_engine.cpp
    src/damage.cpp
    src/ai.cpp
    src/ai_context.cpp
    src/ai_vm.cpp
    src/ai_scripts.cpp
    src/factory.cpp
    src/data/species_data.cpp
    src/data/move_data.cpp
    src/data/type_chart.cpp
    src/data/frontier_mons.cpp
)

target_include_directories(battle_sim PUBLIC include)
target_compile_options(battle_sim PRIVATE -Wall -Wextra -O3 -fPIC)

# Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(pybattle_native src/python/bindings.cpp)
    target_link_libraries(pybattle_native PRIVATE battle_sim)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_damage tests/test_damage.cpp)
    target_link_libraries(test_damage battle_sim)
    add_test(NAME DamageTests COMMAND test_damage)
    
    add_executable(test_battle tests/test_battle.cpp)
    target_link_libraries(test_battle battle_sim)
    add_test(NAME BattleTests COMMAND test_battle)

    add_executable(test_factory tests/test_factory.cpp)
    target_link_libraries(test_factory battle_sim)
    add_test(NAME FactoryTests COMMAND test_factory)

    add_executable(test_ai tests/test_ai.cpp)
    target_link_libraries(test_ai battle_sim)
    add_test(NAME AITests COMMAND test_ai)
endif()
