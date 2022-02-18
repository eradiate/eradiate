message(STATUS "Building Eradiate plugins")

# Function for creating Mitsuba plugins
function(add_plugin)
  list(GET ARGV 0 TARGET)
  list(REMOVE_AT ARGV 0)
  add_library(${TARGET} SHARED ${ARGV})
  target_link_libraries(${TARGET} PRIVATE mitsuba)
  set_target_properties(${TARGET} PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${ERT_MI_BINARY_DIR}/plugins
    RUNTIME_OUTPUT_DIRECTORY ${ERT_MI_BINARY_DIR}/plugins
    FOLDER plugins/${MI_PLUGIN_PREFIX}/${TARGET}
  )
  install(
    TARGETS ${TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}/plugins
  )
endfunction(add_plugin)

# Include Mitsuba dirs
# Note: Definitely not nice CMake work, can be improved
include_directories(
  ${ERT_MI_DIR}/include
  ${ERT_MI_DIR}/ext/tinyformat
  ${ERT_MI_BINARY_DIR}/include  # For config.h
)

# Recurse into sources
add_subdirectory(src)
