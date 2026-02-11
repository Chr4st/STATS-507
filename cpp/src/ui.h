#pragma once

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

namespace geoag {

class AppState;

/// Build the FTXUI component tree for the terminal UI
ftxui::Component BuildUI(AppState& state, ftxui::ScreenInteractive& screen);

}  // namespace geoag
