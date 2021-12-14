# ---------------------------------------------------------
# Top-level Eradiate CMake file, requires a recent version.
# ---------------------------------------------------------

cmake_minimum_required (VERSION 3.13...3.18)

project(mitsuba
  DESCRIPTION
    "Eradiate Plugins"
  LANGUAGES
    CXX C
)

# Build dependencies
add_subdirectory(ext)

# Build plugins
# Note: Not a very nice CMake setup, likely this could be done better
get_filename_component(ERT_MITSUBA_DIR "ext/mitsuba2" ABSOLUTE)
get_filename_component(ERT_MITSUBA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ext/mitsuba2" ABSOLUTE)
add_subdirectory(src/plugins)
