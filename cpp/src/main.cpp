/**
 * GeoAg Arb Terminal — C++20 FTXUI Terminal Client
 *
 * Connects to the Python API server via WebSocket, receives macro indicators,
 * region nowcasts, and trade ideas, and renders them in a real-time TUI.
 *
 * DISCLAIMER: For research only; not investment advice.
 */

#include "state.h"
#include "ui.h"
#include "ws_client.h"

#include <ftxui/component/screen_interactive.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

static void print_usage() {
    std::cout << "GeoAg Arb Terminal v0.1.0\n"
              << "Usage: ./terminal [OPTIONS]\n"
              << "  --url URL    WebSocket server URL (default: ws://localhost:8777/ws)\n"
              << "  --help       Show this help\n"
              << "\nDisclaimer: For research only; not investment advice.\n";
}

int main(int argc, char* argv[]) {
    std::string ws_url = "ws://localhost:8777/ws";

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--url" && i + 1 < argc) {
            ws_url = argv[++i];
        }
    }

    // Create shared state
    geoag::AppState state;

    // Start WebSocket client in background
    geoag::WsClient ws_client(state, ws_url);
    ws_client.start();

    // Create FTXUI screen
    auto screen = ftxui::ScreenInteractive::Fullscreen();

    // Build UI component
    auto ui = geoag::BuildUI(state, screen);

    // Periodic screen refresh (1 second)
    std::atomic<bool> running{true};
    std::thread refresh_thread([&screen, &running] {
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            screen.Post(ftxui::Event::Custom);
        }
    });

    // Run the UI (blocks until quit)
    screen.Loop(ui);

    // Cleanup
    running = false;
    ws_client.stop();
    if (refresh_thread.joinable()) {
        refresh_thread.join();
    }

    std::cout << "\nGeoAg Arb Terminal closed.\n"
              << "Disclaimer: For research only; not investment advice.\n";
    return 0;
}
