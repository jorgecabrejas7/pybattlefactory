cmake_minimum_required(VERSION 3.16)
project(pokemon_battle_simulator VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Web build (emscripten, scripts/build_wasm.sh): only the gen3 core and the web targets.
if(EMSCRIPTEN)
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_EMU_HOST OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()

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
    src/web/factory_session.cpp
    src/gen3/generated/stubs.c
    src/gen3/generated/rom_script_tables.c
    src/gen3/generated/rom_layout_check.c
    src/gen3/generated/rom_scripts_data.c
    src/gen3/generated/rom_ram_map.c
)
set_target_properties(gen3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gen3 PRIVATE GEN3_HOST)
target_include_directories(gen3 PUBLIC include PRIVATE src)
target_compile_options(gen3 PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-std=gnu17 -w -O2 -g -funsigned-char -fwrapv -fno-strict-aliasing -ftrivial-auto-var-init=zero>
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/third_party/pokeemerald/include>"
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/third_party/pokeemerald/src>"
    "$<$<COMPILE_LANGUAGE:C>:SHELL:-iquote ${CMAKE_SOURCE_DIR}/src>")
if(EMSCRIPTEN)
    target_compile_options(gen3 PRIVATE -fwasm-exceptions)
endif()

# Determinism smoke test: native and wasm builds must print identical digests.
option(BUILD_WEB_SMOKE "Build the gen3 determinism smoke test" ON)
if(BUILD_WEB_SMOKE)
    add_executable(gen3_smoke src/web/smoke_main.cpp)
    target_link_libraries(gen3_smoke PRIVATE gen3)
    if(EMSCRIPTEN)
        target_compile_options(gen3_smoke PRIVATE -fwasm-exceptions)
        target_link_options(gen3_smoke PRIVATE -fwasm-exceptions -sALLOW_MEMORY_GROWTH -sENVIRONMENT=node -sSTACK_SIZE=1MB)
    endif()
endif()

# Browser build: the C API of src/web/web_api.cpp as an ES module + wasm (scripts/build_wasm.sh gen3_web)
if(EMSCRIPTEN)
    add_executable(gen3_web src/web/web_api.cpp)
    target_link_libraries(gen3_web PRIVATE gen3)
    target_compile_options(gen3_web PRIVATE -fwasm-exceptions)
    set_target_properties(gen3_web PROPERTIES SUFFIX ".mjs")
    target_link_options(gen3_web PRIVATE -fwasm-exceptions -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createGen3
        -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB -sFILESYSTEM=0
        "-sEXPORTED_FUNCTIONS=_malloc,_free"
        "-sEXPORTED_RUNTIME_METHODS=UTF8ToString,HEAPU8,HEAP32,HEAPF32")
endif()

# The real game for the browser: libmgba + emu/web/gba_web.cpp (scripts/build_wasm.sh gba_web)
if(EMSCRIPTEN)
    include(FetchContent)
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
    set(USE_ZLIB OFF CACHE BOOL "" FORCE)
    set(USE_PNG OFF CACHE BOOL "" FORCE)
    set(USE_SQLITE3 OFF CACHE BOOL "" FORCE)
    set(USE_EPOXY OFF CACHE STRING "" FORCE)
    set(BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
    FetchContent_GetProperties(mgba)
    if(NOT mgba_POPULATED)
        FetchContent_Populate(mgba)
        set(_c_flags "${CMAKE_C_FLAGS}")
        string(APPEND CMAKE_C_FLAGS " -D_GNU_SOURCE")          # mGBA only adds it for Linux
        add_subdirectory(${mgba_SOURCE_DIR} ${mgba_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(CMAKE_C_FLAGS "${_c_flags}")
    endif()
    add_executable(gba_web emu/web/gba_web.cpp)
    target_link_libraries(gba_web PRIVATE mgba)
    set_target_properties(gba_web PROPERTIES SUFFIX ".mjs")
    target_link_options(gba_web PRIVATE -O3 -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createGba
        -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=1MB -sFILESYSTEM=0
        "-sEXPORTED_FUNCTIONS=_malloc,_free"
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP16,HEAP32")
endif()

# A Nintendo DS for the browser: DeSmuME (the emscripten fork github.com/44670/desmume-wasm, GPL-2.0, a local clone)
# + emu/web/nds_web.cpp (scripts/build_wasm.sh nds_web; the script mounts the clone at /desmume in Docker).
# Core only: interpreter (no JIT), software 3D, no OpenGL/lua/gdbstub/frontends. Three fork bugs and the clock hook are
# patched on copies of five sources in the build tree (the clone itself is never modified): see _nds_patch below.
set(DESMUME_WASM_DIR "/home/apollo/Dev/desmume-wasm" CACHE PATH "desmume-wasm clone (github.com/44670/desmume-wasm)")
if(EMSCRIPTEN AND EXISTS "${DESMUME_WASM_DIR}/desmume/src/NDSSystem.cpp")
    set(_ds "${DESMUME_WASM_DIR}/desmume/src")
    set(_nds_srcs
        armcpu.cpp arm_instructions.cpp bios.cpp cheatSystem.cpp common.cpp cp15.cpp Database.cpp debug.cpp
        driver.cpp emufile.cpp encrypt.cpp FIFO.cpp firmware.cpp gfx3d.cpp GPU.cpp matrix.cpp mc.cpp mic.cpp MMU.cpp
        movie.cpp NDSSystem.cpp path.cpp rasterize.cpp readwrite.cpp render3D.cpp ROMReader.cpp rtc.cpp saves.cpp
        slot1.cpp slot2.cpp SPU.cpp texcache.cpp thumb_instructions.cpp version.cpp wifi.cpp
        addons/slot1comp_mc.cpp addons/slot1comp_protocol.cpp addons/slot1comp_rom.cpp addons/slot1_none.cpp
        addons/slot1_r4.cpp addons/slot1_retail_auto.cpp addons/slot1_retail_mcrom.cpp
        addons/slot1_retail_mcrom_debug.cpp addons/slot1_retail_nand.cpp addons/slot2_auto.cpp
        addons/slot2_expMemory.cpp addons/slot2_gbagame.cpp addons/slot2_guitarGrip.cpp addons/slot2_mpcf.cpp
        addons/slot2_none.cpp addons/slot2_paddle.cpp addons/slot2_passme.cpp addons/slot2_piano.cpp
        addons/slot2_rumblepak.cpp
        metaspu/metaspu.cpp filter/deposterize.cpp filter/xbrz.cpp
        utils/advanscene.cpp utils/colorspacehandler/colorspacehandler.cpp utils/datetime.cpp
        utils/decrypt/crc.cpp utils/decrypt/decrypt.cpp utils/decrypt/header.cpp utils/dlditool.cpp
        utils/emufat.cpp utils/fsnitro.cpp utils/guid.cpp utils/vfat.cpp utils/xstring.cpp
        utils/libfat/cache.cpp utils/libfat/directory.cpp utils/libfat/disc.cpp utils/libfat/fatdir.cpp
        utils/libfat/fatfile.cpp utils/libfat/file_allocation_table.cpp utils/libfat/filetime.cpp
        utils/libfat/libfat.cpp utils/libfat/libfat_public_api.cpp utils/libfat/lock.cpp utils/libfat/partition.cpp
        utils/tinyxml/tinystr.cpp utils/tinyxml/tinyxml.cpp utils/tinyxml/tinyxmlerror.cpp
        utils/tinyxml/tinyxmlparser.cpp
        libretro-common/compat/compat_strl.c libretro-common/compat/compat_getopt.c
        libretro-common/file/file_path.c libretro-common/file/retro_dirent.c libretro-common/file/retro_stat.c
        libretro-common/rthreads/async_job.c
        libretro-common/rthreads/rsemaphore.c libretro-common/rthreads/rthreads.c
        libretro-common/encodings/encoding_utf.c libretro-common/string/stdstring.c)
    # _nds_patch(<file> <from> <to> [<from> <to> ...]): compile a patched copy of <file>; every <from> must occur;
    # write ';' as <SC>.
    set(_nds_patched_dir "${CMAKE_BINARY_DIR}/nds_patched")
    set(_nds_patched)
    function(_nds_patch file)
        file(READ "${_ds}/${file}" _txt)
        set(_args ${ARGN})
        list(LENGTH _args _n)
        math(EXPR _last "${_n} - 1")
        foreach(_i RANGE 0 ${_last} 2)
            math(EXPR _j "${_i} + 1")
            list(GET _args ${_i} _from)
            list(GET _args ${_j} _to)
            string(REPLACE "<SC>" ";" _from "${_from}")   # list arguments cannot hold ';'
            string(REPLACE "<SC>" ";" _to "${_to}")
            string(FIND "${_txt}" "${_from}" _pos)
            if(_pos EQUAL -1)
                message(FATAL_ERROR "nds_web: patch for ${file} no longer applies (missing: ${_from})")
            endif()
            string(REPLACE "${_from}" "${_to}" _txt "${_txt}")
        endforeach()
        file(WRITE "${_nds_patched_dir}/${file}.new" "${_txt}")
        configure_file("${_nds_patched_dir}/${file}.new" "${_nds_patched_dir}/${file}" COPYONLY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ds}/${file}")
        list(REMOVE_ITEM _nds_srcs ${file})
        set(_nds_srcs ${_nds_srcs} PARENT_SCOPE)
        list(APPEND _nds_patched "${_nds_patched_dir}/${file}")
        set(_nds_patched ${_nds_patched} PARENT_SCOPE)
    endfunction()
    # mc.cpp: the backup device deletes its file when destroyed, but in the fork that file is the global savFile and
    # MMU_Reset rebuilds the device on every reset (use-after-free). Never delete savFile.
    _nds_patch(mc.cpp
        "BackupDevice::~BackupDevice()\n{\n\tdelete fpMC<SC>"
        "extern EMUFILE_MEMORY* savFile<SC>\nBackupDevice::~BackupDevice()\n{\n\tif ((void*)fpMC != (void*)savFile) delete fpMC<SC>"
        "void BackupDevice::close_rom()\n{\n\tfpMC->fflush()<SC>\n\tdelete fpMC<SC>"
        "void BackupDevice::close_rom()\n{\n\tfpMC->fflush()<SC>\n\tif ((void*)fpMC != (void*)savFile) delete fpMC<SC>")
    # rtc.cpp: the clock comes from nds_web_rtc_now() (wall clock, or nds_set_rtc's emulated-time clock).
    _nds_patch(rtc.cpp
        "DateTime rtcGetTime(void)\n{"
        "DateTime nds_web_rtc_now()<SC>\nDateTime rtcGetTime(void)\n{"
        "return DateTime::get_Now().AddHours(rtcHourOverride)<SC>"
        "return nds_web_rtc_now().AddHours(rtcHourOverride)<SC>")
    # ROMReader.cpp: a read starting past the end of the ROM memcpy'd a negative length (out-of-bounds on tiny files).
    _nds_patch(ROMReader.cpp
        "read = romLen - pos<SC>"
        "read = pos < romLen ? romLen - pos : 0<SC>")
    # slot1.cpp/slot2.cpp: the fork leaves some device slots NULL (R4, debug, most slot-2 devices) but the
    # by-ID lookup used when loading a savestate dereferences every slot (null call -> savestates never loaded).
    _nds_patch(slot1.cpp
        "if (slot1_List[i]->info()->id() == ID)"
        "if (slot1_List[i] && slot1_List[i]->info()->id() == ID)")
    _nds_patch(slot2.cpp
        "if (slot2_List[i]->info()->id() == ID)"
        "if (slot2_List[i] && slot2_List[i]->info()->id() == ID)")
    list(TRANSFORM _nds_srcs PREPEND "${_ds}/")
    add_executable(nds_web emu/web/nds_web.cpp ${_nds_srcs} ${_nds_patched})
    target_include_directories(nds_web PRIVATE "${_ds}" "${_ds}/libretro-common/include")
    target_compile_definitions(nds_web PRIVATE HAVE_LIBZ _GNU_SOURCE)
    set(NDS_WEB_OPT "-O3" CACHE STRING "nds_web optimisation flags (e.g. \"-O1 -g\" to debug)")
    separate_arguments(_nds_o UNIX_COMMAND "${NDS_WEB_OPT}")
    set(_nds_opt ${_nds_o} -msimd128 -sUSE_ZLIB=1 -fno-strict-aliasing)
    target_compile_options(nds_web PRIVATE ${_nds_opt} -w)
    set_target_properties(nds_web PROPERTIES SUFFIX ".mjs")
    target_link_options(nds_web PRIVATE ${_nds_opt} -sMODULARIZE -sEXPORT_ES6 -sEXPORT_NAME=createNds
        -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=128MB -sMAXIMUM_MEMORY=2GB
        -sSTACK_SIZE=1MB
        "-sEXPORTED_FUNCTIONS=_malloc,_free"
        "-sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP16,HEAP32")
endif()

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

    pybind11_add_module(pybattle_native src/python/bindings.cpp src/python/search_bindings.cpp src/python/observer_bindings.cpp
        src/python/web_bindings.cpp)
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
