#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace geoag {

class AppState;

/// WebSocket client with auto-reconnect
class WsClient {
public:
    explicit WsClient(AppState& state, const std::string& url = "ws://localhost:8777/ws");
    ~WsClient();

    void start();
    void stop();
    bool is_running() const { return running_.load(); }

private:
    void run_loop();
    void handle_message(const std::string& msg);
    void parse_macro(const std::string& data_str);
    void parse_regions(const std::string& data_str);
    void parse_trade_ideas(const std::string& data_str);

    AppState& state_;
    std::string url_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::thread thread_;
};

}  // namespace geoag
