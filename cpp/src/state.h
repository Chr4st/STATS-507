#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace geoag {

// ---------------------------------------------------------------------------
// Data structures matching the Python API JSON schemas
// ---------------------------------------------------------------------------

struct CatalystEvent {
    std::string date;
    std::string name;
    std::string agency;
    std::string impact;
    std::vector<std::string> commodities;
    int days_until = 0;
};

struct MacroData {
    std::string timestamp;
    double global_crop_stress_nowcast = 0.0;
    double export_risk_index = 0.0;
    double food_inflation_pressure_proxy = 0.0;
    std::vector<CatalystEvent> catalysts;
    // Previous values for delta display
    double prev_global_crop_stress = 0.0;
    double prev_export_risk = 0.0;
    double prev_food_inflation = 0.0;
};

struct RegionData {
    std::string region_id;
    std::string region_name;
    std::string timestamp;
    double stress_index = 0.0;
    double growth_index = 0.0;
    double yield_shock_mean = 0.0;
    double yield_shock_sigma = 0.0;
    double confidence = 0.0;
    std::vector<std::string> drivers;
};

struct InstrumentLeg {
    std::string symbol;
    std::string direction;
    double weight = 1.0;
    std::string instrument_name;
};

struct TradeIdea {
    std::string id;
    std::string timestamp;
    std::string trade_type;
    std::vector<InstrumentLeg> instruments;
    std::string rationale;
    double expected_edge = 0.0;
    std::vector<std::string> risk_notes;
    std::vector<std::string> risk_tags;
    bool tradable_now = false;
    std::string best_window;
    std::vector<std::string> hedges;
    double pin_risk_score = 0.0;
    double confidence = 0.0;
};

// ---------------------------------------------------------------------------
// Application state (thread-safe)
// ---------------------------------------------------------------------------

class AppState {
public:
    void set_macro(const MacroData& macro) {
        std::lock_guard<std::mutex> lock(mtx_);
        // Save previous values for delta
        macro_data_.prev_global_crop_stress = macro_data_.global_crop_stress_nowcast;
        macro_data_.prev_export_risk = macro_data_.export_risk_index;
        macro_data_.prev_food_inflation = macro_data_.food_inflation_pressure_proxy;
        macro_data_ = macro;
        macro_data_.prev_global_crop_stress = macro_data_.prev_global_crop_stress;
        macro_data_.prev_export_risk = macro_data_.prev_export_risk;
        macro_data_.prev_food_inflation = macro_data_.prev_food_inflation;
        last_update_ = std::chrono::system_clock::now();
    }

    MacroData get_macro() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return macro_data_;
    }

    void set_regions(const std::vector<RegionData>& regions) {
        std::lock_guard<std::mutex> lock(mtx_);
        regions_ = regions;
        last_update_ = std::chrono::system_clock::now();
    }

    std::vector<RegionData> get_regions() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return regions_;
    }

    void set_trade_ideas(const std::vector<TradeIdea>& ideas) {
        std::lock_guard<std::mutex> lock(mtx_);
        trade_ideas_ = ideas;
        last_update_ = std::chrono::system_clock::now();
    }

    std::vector<TradeIdea> get_trade_ideas() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return trade_ideas_;
    }

    void set_connected(bool connected) {
        std::lock_guard<std::mutex> lock(mtx_);
        connected_ = connected;
    }

    bool is_connected() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return connected_;
    }

    void set_selected_idea(int idx) {
        std::lock_guard<std::mutex> lock(mtx_);
        selected_idea_ = idx;
    }

    int get_selected_idea() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return selected_idea_;
    }

    void set_active_tab(int tab) {
        std::lock_guard<std::mutex> lock(mtx_);
        active_tab_ = tab;
    }

    int get_active_tab() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return active_tab_;
    }

    void set_show_detail(bool show) {
        std::lock_guard<std::mutex> lock(mtx_);
        show_detail_ = show;
    }

    bool get_show_detail() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return show_detail_;
    }

    std::string get_last_update_str() const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - last_update_).count();
        if (diff < 2) return "just now";
        return std::to_string(diff) + "s ago";
    }

private:
    mutable std::mutex mtx_;
    MacroData macro_data_;
    std::vector<RegionData> regions_;
    std::vector<TradeIdea> trade_ideas_;
    bool connected_ = false;
    int selected_idea_ = -1;
    int active_tab_ = 0;  // 0=all, 1=macro, 2=regions, 3=trades
    bool show_detail_ = false;
    std::chrono::system_clock::time_point last_update_ = std::chrono::system_clock::now();
};

}  // namespace geoag
