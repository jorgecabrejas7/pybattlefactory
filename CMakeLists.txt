cmake_minimum_required(VERSION 3.16)
project(pokemon_battle_simulator VERSION 1.0 LANGUAGES C CXX ASM)

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
    src/data/item_data.cpp
    src/data/factory_tables.cpp
)

target_include_directories(battle_sim PUBLIC include)

# gen3 battle core: the game's own battle code (vendored by scripts/sync_decomp.py) plus the
# headless host layer. Compiled as C with GBA-like semantics (unsigned char, wrapping ints).
file(GLOB GEN3_VENDOR_SOURCES ${CMAKE_SOURCE_DIR}/third_party/pokeemerald/src/*.c)
add_library(gen3 STATIC
    ${GEN3_VENDOR_SOURCES}
    src/gen3/host.c
    src/gen3/factory_run.c
    src/gen3/search_host.c
    src/gen3/observer_host.c
    src/gen3/observer.cpp
    src/gen3/game.cpp
    src/gen3/mcts.cpp
    src/gen3/search.cpp
    src/gen3/generated/stubs.c
    src/gen3/generated/rom_script_tables.c
    src/gen3/generated/rom_layout_check.c
    src/gen3/generated/rom_scripts.S
    src/gen3/generated/rom_ram_map.S
)
set_target_properties(gen3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gen3 PRIVATE GEN3_HOST)
target_include_directories(gen3 PUBLIC include PRIVATE src)
target_compile_options(gen3 PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-std=gnu17 -w -O2 -g -funsigned-char -fwrapv -fno-strict-aliasing -ftrivial-auto-var-init=zero>
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/third_party/pokeemerald/include>"
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/third_party/pokeemerald/src>"
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/src>"
    $<$<COMPILE_LANGUAGE:ASM>:-Wa,-I${CMAKE_SOURCE_DIR}/src/gen3/generated>)
set_source_files_properties(src/gen3/generated/rom_scripts.S PROPERTIES
    OBJECT_DEPENDS ${CMAKE_SOURCE_DIR}/src/gen3/generated/rom_scripts.bin)
if(MSVC)
    target_compile_options(battle_sim PRIVATE /W3 /O2)
else()
    target_compile_options(battle_sim PRIVATE -Wall -Wextra -O3 -fPIC)
endif()

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

    pybind11_add_module(pybattle_native src/python/bindings.cpp src/python/search_bindings.cpp src/python/observer_bindings.cpp)
    target_link_libraries(pybattle_native PRIVATE battle_sim gen3)
    if(NOT MSVC)
        target_link_options(pybattle_native PRIVATE -static-libgcc -static-libstdc++)
    endif()
    # Dev builds drop the module next to the Python package so `import pybattle` works in-tree.
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(pybattle_native PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pybattle)
    endif()
endif()

# Headless emulator host: minimal static libmgba 0.10.2 (same core as mgba-qt 0.10.2),
# built without frontends/ffmpeg/etc. so it has no system library dependencies.
option(BUILD_EMU_HOST "Build the headless mGBA host module" ON)
if(BUILD_PYTHON_BINDINGS AND BUILD_EMU_HOST)
    FetchContent_Declare(
        mgba
        GIT_REPOSITORY https://github.com/mgba-emu/mgba.git
        GIT_TAG        0.10.2
        GIT_SHALLOW    TRUE
    )
    set(LIBMGBA_ONLY ON CACHE BOOL "" FORCE)
    set(M_CORE_GB OFF CACHE BOOL "" FORCE)
    set(ENABLE_SCRIPTING OFF CACHE BOOL "" FORCE)
    set(USE_LUA OFF CACHE BOOL "" FORCE)
    set(USE_ELF OFF CACHE BOOL "" FORCE)
    set(USE_LZMA OFF CACHE BOOL "" FORCE)
    set(USE_EPOXY OFF CACHE STRING "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_GetProperties(mgba)
    if(NOT mgba_POPULATED)
        FetchContent_Populate(mgba)
        add_subdirectory(${mgba_SOURCE_DIR} ${mgba_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    pybind11_add_module(pybattle_emu emu/host/emulator.cpp)
    target_link_libraries(pybattle_emu PRIVATE mgba)
    if(NOT MSVC)
        target_link_options(pybattle_emu PRIVATE -static-libgcc -static-libstdc++)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(pybattle_emu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pybattle)
    endif()
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

    add_executable(test_items tests/test_items.cpp)
    target_link_libraries(test_items battle_sim)
    add_test(NAME ItemTests COMMAND test_items)

    add_executable(test_ai tests/test_ai.cpp)
    target_link_libraries(test_ai battle_sim)
    add_test(NAME AITests COMMAND test_ai)
endif()
