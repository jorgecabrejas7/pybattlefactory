# CMake generated Testfile for 
# Source directory: /home/apollo/Dev/pybattlefactory
# Build directory: /home/apollo/Dev/pybattlefactory/build_tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(DamageTests "/home/apollo/Dev/pybattlefactory/build_tests/test_damage")
set_tests_properties(DamageTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pybattlefactory/CMakeLists.txt;53;add_test;/home/apollo/Dev/pybattlefactory/CMakeLists.txt;0;")
add_test(BattleTests "/home/apollo/Dev/pybattlefactory/build_tests/test_battle")
set_tests_properties(BattleTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pybattlefactory/CMakeLists.txt;57;add_test;/home/apollo/Dev/pybattlefactory/CMakeLists.txt;0;")
add_test(FactoryTests "/home/apollo/Dev/pybattlefactory/build_tests/test_factory")
set_tests_properties(FactoryTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pybattlefactory/CMakeLists.txt;61;add_test;/home/apollo/Dev/pybattlefactory/CMakeLists.txt;0;")
add_test(ItemTests "/home/apollo/Dev/pybattlefactory/build_tests/test_items")
set_tests_properties(ItemTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pybattlefactory/CMakeLists.txt;65;add_test;/home/apollo/Dev/pybattlefactory/CMakeLists.txt;0;")
add_test(AITests "/home/apollo/Dev/pybattlefactory/build_tests/test_ai")
set_tests_properties(AITests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pybattlefactory/CMakeLists.txt;69;add_test;/home/apollo/Dev/pybattlefactory/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
