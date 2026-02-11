cmake_minimum_required(VERSION 3.20)
project(geoag_terminal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# FTXUI — Terminal UI library
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG        v5.0.0
    GIT_SHALLOW    TRUE
)

# IXWebSocket — WebSocket client
FetchContent_Declare(
    ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.5
    GIT_SHALLOW    TRUE
)

# nlohmann/json — JSON parser
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
set(USE_TLS OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ftxui ixwebsocket nlohmann_json)

# ---------------------------------------------------------------------------
# Terminal executable
# ---------------------------------------------------------------------------
add_executable(terminal
    src/main.cpp
    src/ui.cpp
    src/ws_client.cpp
)

target_include_directories(terminal PRIVATE src)

target_link_libraries(terminal PRIVATE
    ftxui::screen
    ftxui::dom
    ftxui::component
    ixwebsocket
    nlohmann_json::nlohmann_json
)

# Install
install(TARGETS terminal DESTINATION bin)
