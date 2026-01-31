# CMake generated Testfile for 
# Source directory: /home/apollo/Dev/pokeemerald/simulator
# Build directory: /home/apollo/Dev/pokeemerald/simulator/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(DamageTests "/home/apollo/Dev/pokeemerald/simulator/build/test_damage")
set_tests_properties(DamageTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;48;add_test;/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;0;")
add_test(BattleTests "/home/apollo/Dev/pokeemerald/simulator/build/test_battle")
set_tests_properties(BattleTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;52;add_test;/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;0;")
add_test(FactoryTests "/home/apollo/Dev/pokeemerald/simulator/build/test_factory")
set_tests_properties(FactoryTests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;56;add_test;/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;0;")
add_test(AITests "/home/apollo/Dev/pokeemerald/simulator/build/test_ai")
set_tests_properties(AITests PROPERTIES  _BACKTRACE_TRIPLES "/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;60;add_test;/home/apollo/Dev/pokeemerald/simulator/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
