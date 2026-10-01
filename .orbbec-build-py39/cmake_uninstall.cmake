if(NOT EXISTS "/home/medpal/2026medpal/.orbbec-build-py39/install_manifest.txt")
    message(FATAL_ERROR "Cannot find install manifest: /home/medpal/2026medpal/.orbbec-build-py39/install_manifest.txt")
endif()

file(READ "/home/medpal/2026medpal/.orbbec-build-py39/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
    message(STATUS "Uninstalling: $ENV{DESTDIR}${file}")
    if(EXISTS "$ENV{DESTDIR}${file}")
        execute_process(
            COMMAND /usr/bin/cmake -E remove "$ENV{DESTDIR}${file}"
            OUTPUT_VARIABLE rm_out
            RESULT_VARIABLE rm_retval
        )
        if(NOT ${rm_retval} EQUAL 0)
            message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
        endif()
    else()
        message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
    endif()
endforeach()
