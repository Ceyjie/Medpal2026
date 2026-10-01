# CMake generated Testfile for 
# Source directory: /home/medpal/pyorbbecsdk_v1
# Build directory: /home/medpal/2026medpal/.orbbec-build-py39
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_context "/home/medpal/2026medpal/coral_venv/bin/python3" "-v" "/home/medpal/pyorbbecsdk_v1/test/test_context.py")
set_tests_properties(test_context PROPERTIES  _BACKTRACE_TRIPLES "/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;90;add_test;/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;0;")
add_test(test_device "/home/medpal/2026medpal/coral_venv/bin/python3" "-v" "/home/medpal/pyorbbecsdk_v1/test/test_device.py")
set_tests_properties(test_device PROPERTIES  _BACKTRACE_TRIPLES "/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;90;add_test;/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;0;")
add_test(test_pipeline "/home/medpal/2026medpal/coral_venv/bin/python3" "-v" "/home/medpal/pyorbbecsdk_v1/test/test_pipeline.py")
set_tests_properties(test_pipeline PROPERTIES  _BACKTRACE_TRIPLES "/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;90;add_test;/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;0;")
add_test(test_sensor_control "/home/medpal/2026medpal/coral_venv/bin/python3" "-v" "/home/medpal/pyorbbecsdk_v1/test/test_sensor_control.py")
set_tests_properties(test_sensor_control PROPERTIES  _BACKTRACE_TRIPLES "/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;90;add_test;/home/medpal/pyorbbecsdk_v1/CMakeLists.txt;0;")
