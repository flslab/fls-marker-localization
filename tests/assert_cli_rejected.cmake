if(NOT DEFINED EYE_EXECUTABLE OR NOT DEFINED CLI_OPTION OR
   NOT DEFINED EXPECTED_PATTERN)
    message(FATAL_ERROR "CLI rejection test is missing a required parameter")
endif()

set(command "${EYE_EXECUTABLE}" "${CLI_OPTION}")
if(DEFINED CLI_VALUE)
    list(APPEND command "${CLI_VALUE}")
endif()

execute_process(
    COMMAND ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(result EQUAL 0)
    message(FATAL_ERROR "${CLI_OPTION} was unexpectedly accepted")
endif()

set(output "${stdout}\n${stderr}")
if(NOT output MATCHES "${EXPECTED_PATTERN}")
    message(FATAL_ERROR
        "${CLI_OPTION} failed without the expected diagnostic. Output:\n${output}"
    )
endif()
