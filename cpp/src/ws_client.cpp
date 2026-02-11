#include "ws_client.h"
#include "state.h"

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <thread>

using json = nlohmann::json;

namespace geoag {

WsClient::WsClient(AppState& state, const std::string& url)
    : state_(state), url_(url) {
    ix::initNetSystem();
}

WsClient::~WsClient() {
    stop();
    ix::uninitNetSystem();
}

void WsClient::start() {
    should_stop_ = false;
    running_ = true;
    thread_ = std::thread(&WsClient::run_loop, this);
}

void WsClient::stop() {
    should_stop_ = true;
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void WsClient::run_loop() {
    int backoff_ms = 1000;
    constexpr int max_backoff_ms = 30000;

    while (!should_stop_) {
        ix::WebSocket ws;
        ws.setUrl(url_);
        ws.setHandshakeTimeout(5);
        ws.setPingInterval(10);

        std::atomic<bool> ws_open{false};

        ws.setOnMessageCallback([&](const ix::WebSocketMessagePtr& msg) {
            if (msg->type == ix::WebSocketMessageType::Open) {
                state_.set_connected(true);
                ws_open = true;
                backoff_ms = 1000;  // Reset backoff on success
            } else if (msg->type == ix::WebSocketMessageType::Close) {
                state_.set_connected(false);
                ws_open = false;
            } else if (msg->type == ix::WebSocketMessageType::Error) {
                state_.set_connected(false);
                ws_open = false;
            } else if (msg->type == ix::WebSocketMessageType::Message) {
                handle_message(msg->str);
            }
        });

        ws.start();

        // Wait for connection or shutdown
        while (!should_stop_ && !ws_open) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Stay connected until error or shutdown
        while (!should_stop_ && ws_open) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        ws.stop();
        state_.set_connected(false);

        if (!should_stop_) {
            // Reconnect with backoff
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, max_backoff_ms);
        }
    }

    running_ = false;
}

void WsClient::handle_message(const std::string& msg) {
    try {
        auto j = json::parse(msg);
        std::string msg_type = j.value("type", "");

        if (msg_type == "macro") {
            parse_macro(j["data"].dump());
        } else if (msg_type == "regions") {
            parse_regions(j["data"].dump());
        } else if (msg_type == "trade_ideas") {
            parse_trade_ideas(j["data"].dump());
        } else if (msg_type == "heartbeat") {
            // Just confirms connection is alive
        }
    } catch (const std::exception& e) {
        // Silently ignore parse errors — don't crash terminal
    }
}

void WsClient::parse_macro(const std::string& data_str) {
    try {
        auto j = json::parse(data_str);
        MacroData macro;
        macro.timestamp = j.value("timestamp", "—");
        macro.global_crop_stress_nowcast = j.value("global_crop_stress_nowcast", 0.0);
        macro.export_risk_index = j.value("export_risk_index", 0.0);
        macro.food_inflation_pressure_proxy = j.value("food_inflation_pressure_proxy", 0.0);

        if (j.contains("volatility_catalyst_calendar")) {
            for (const auto& c : j["volatility_catalyst_calendar"]) {
                CatalystEvent ev;
                ev.date = c.value("date", "");
                ev.name = c.value("name", "");
                ev.agency = c.value("agency", "");
                ev.impact = c.value("impact", "");
                ev.days_until = c.value("days_until", 0);
                if (c.contains("commodities")) {
                    for (const auto& cm : c["commodities"]) {
                        ev.commodities.push_back(cm.get<std::string>());
                    }
                }
                macro.catalysts.push_back(ev);
            }
        }

        state_.set_macro(macro);
    } catch (...) {}
}

void WsClient::parse_regions(const std::string& data_str) {
    try {
        auto j = json::parse(data_str);
        std::vector<RegionData> regions;

        for (const auto& r : j) {
            RegionData rd;
            rd.region_id = r.value("region_id", "—");
            rd.region_name = r.value("region_name", "—");
            rd.timestamp = r.value("timestamp", "—");
            rd.stress_index = r.value("stress_index", 0.0);
            rd.growth_index = r.value("growth_index", 0.0);
            rd.yield_shock_mean = r.value("yield_shock_mean", 0.0);
            rd.yield_shock_sigma = r.value("yield_shock_sigma", 0.0);
            rd.confidence = r.value("confidence", 0.0);

            if (r.contains("drivers")) {
                for (const auto& d : r["drivers"]) {
                    rd.drivers.push_back(d.get<std::string>());
                }
            }
            regions.push_back(rd);
        }

        state_.set_regions(regions);
    } catch (...) {}
}

void WsClient::parse_trade_ideas(const std::string& data_str) {
    try {
        auto j = json::parse(data_str);
        std::vector<TradeIdea> ideas;

        for (const auto& t : j) {
            TradeIdea idea;
            idea.id = t.value("id", "—");
            idea.timestamp = t.value("timestamp", "—");
            idea.trade_type = t.value("trade_type", "—");
            idea.rationale = t.value("rationale", "—");
            idea.expected_edge = t.value("expected_edge", 0.0);
            idea.tradable_now = t.value("tradable_now", false);
            idea.best_window = t.value("best_window", "");
            idea.pin_risk_score = t.value("pin_risk_score", 0.0);
            idea.confidence = t.value("confidence", 0.0);

            if (t.contains("instruments")) {
                for (const auto& leg : t["instruments"]) {
                    InstrumentLeg il;
                    il.symbol = leg.value("symbol", "");
                    il.direction = leg.value("direction", "");
                    il.weight = leg.value("weight", 1.0);
                    il.instrument_name = leg.value("instrument_name", "");
                    idea.instruments.push_back(il);
                }
            }

            if (t.contains("risk_notes")) {
                for (const auto& rn : t["risk_notes"]) {
                    idea.risk_notes.push_back(rn.get<std::string>());
                }
            }
            if (t.contains("risk_tags")) {
                for (const auto& rt : t["risk_tags"]) {
                    idea.risk_tags.push_back(rt.get<std::string>());
                }
            }
            if (t.contains("hedges")) {
                for (const auto& h : t["hedges"]) {
                    idea.hedges.push_back(h.get<std::string>());
                }
            }

            ideas.push_back(idea);
        }

        state_.set_trade_ideas(ideas);
    } catch (...) {}
}

}  // namespace geoag
