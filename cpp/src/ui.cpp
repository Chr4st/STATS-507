#include "ui.h"
#include "state.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

using namespace ftxui;

namespace geoag {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string now_local_str() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static std::string fmt_double(double v, int prec = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

static std::string delta_arrow(double current, double previous) {
    double diff = current - previous;
    if (std::abs(diff) < 1e-6) return " —";
    return (diff > 0 ? " ▲" : " ▼") + fmt_double(std::abs(diff), 3);
}

static Element risk_tag_element(const std::string& tag) {
    if (tag == "PIN_RISK") return text(tag) | color(Color::Red);
    if (tag == "SESSION_CLOSED") return text(tag) | color(Color::Yellow);
    if (tag == "LOW_LIQUIDITY") return text(tag) | color(Color::Yellow);
    if (tag == "SPEC_MISMATCH") return text(tag) | color(Color::Magenta);
    return text(tag) | color(Color::White);
}

static Color stress_color(double stress) {
    double a = std::abs(stress);
    if (a > 1.5) return Color::Red;
    if (a > 0.8) return Color::Yellow;
    return Color::Green;
}

static Color confidence_color(double conf) {
    if (conf > 0.7) return Color::Green;
    if (conf > 0.4) return Color::Yellow;
    return Color::Red;
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

static Element render_status_bar(const AppState& state) {
    bool connected = state.is_connected();
    std::string conn_str = connected ? "● CONNECTED" : "○ DISCONNECTED";
    Color conn_color = connected ? Color::Green : Color::Red;

    return hbox({
        text(" GeoAg Arb Terminal ") | bold | color(Color::Cyan),
        separator(),
        text(" " + conn_str + " ") | color(conn_color),
        separator(),
        text(" Updated: " + state.get_last_update_str() + " ") | color(Color::GrayLight),
        separator(),
        text(" " + now_local_str() + " ") | color(Color::GrayLight),
        filler(),
        text(" [q]uit [r]econn [1/2/3]tabs [Enter]detail ") | dim,
    }) | bgcolor(Color::GrayDark);
}

static Element render_macro_panel(const AppState& state) {
    auto macro = state.get_macro();

    Elements rows;
    rows.push_back(text(" MACRO INDICATORS") | bold | color(Color::Cyan));
    rows.push_back(separator());

    auto add_metric = [&](const std::string& label, double val, double prev) {
        Color c = (std::abs(val) > 0.5) ? Color::Yellow : Color::Green;
        rows.push_back(hbox({
            text(" " + label + ": ") | color(Color::White),
            text(fmt_double(val)) | color(c),
            text(delta_arrow(val, prev)) | dim,
        }));
    };

    add_metric("Crop Stress    ", macro.global_crop_stress_nowcast, macro.prev_global_crop_stress);
    add_metric("Export Risk    ", macro.export_risk_index, macro.prev_export_risk);
    add_metric("Inflation Pres.", macro.food_inflation_pressure_proxy, macro.prev_food_inflation);

    rows.push_back(separator());
    rows.push_back(text(" UPCOMING CATALYSTS") | bold | color(Color::Cyan));
    rows.push_back(separator());

    if (macro.catalysts.empty()) {
        rows.push_back(text("  No upcoming catalysts") | dim);
    } else {
        for (const auto& cat : macro.catalysts) {
            Color impact_c = Color::White;
            if (cat.impact == "very_high") impact_c = Color::Red;
            else if (cat.impact == "high") impact_c = Color::Yellow;

            rows.push_back(hbox({
                text("  " + std::to_string(cat.days_until) + "d ") | color(Color::GrayLight),
                text(cat.name) | color(impact_c),
            }));
        }
    }

    return vbox(rows) | border | flex;
}

static Element render_regions_panel(const AppState& state) {
    auto regions = state.get_regions();

    Elements rows;
    rows.push_back(text(" REGIONS") | bold | color(Color::Cyan));
    rows.push_back(separator());

    if (regions.empty()) {
        rows.push_back(text("  Waiting for data...") | dim);
    } else {
        // Header
        rows.push_back(hbox({
            text(" Region              ") | bold | size(WIDTH, EQUAL, 22),
            text("Stress") | bold | size(WIDTH, EQUAL, 8),
            text("Growth") | bold | size(WIDTH, EQUAL, 8),
            text("Conf  ") | bold | size(WIDTH, EQUAL, 7),
            text("Drivers") | bold | flex,
        }));
        rows.push_back(separator());

        for (const auto& r : regions) {
            std::string drivers_str;
            for (size_t i = 0; i < std::min(r.drivers.size(), (size_t)2); ++i) {
                if (i > 0) drivers_str += ", ";
                drivers_str += r.drivers[i];
            }

            rows.push_back(hbox({
                text(" " + r.region_name.substr(0, 20)) | size(WIDTH, EQUAL, 22),
                text(fmt_double(r.stress_index, 3)) | color(stress_color(r.stress_index)) | size(WIDTH, EQUAL, 8),
                text(fmt_double(r.growth_index, 3)) | color(Color::Green) | size(WIDTH, EQUAL, 8),
                text(fmt_double(r.confidence, 2)) | color(confidence_color(r.confidence)) | size(WIDTH, EQUAL, 7),
                text(drivers_str) | dim | flex,
            }));
        }
    }

    return vbox(rows) | border | flex;
}

static Element render_trades_panel(const AppState& state) {
    auto ideas = state.get_trade_ideas();
    int selected = state.get_selected_idea();

    Elements rows;
    rows.push_back(text(" TRADE IDEAS") | bold | color(Color::Cyan));
    rows.push_back(separator());

    if (ideas.empty()) {
        rows.push_back(text("  Waiting for signals...") | dim);
    } else {
        // Header
        rows.push_back(hbox({
            text("#") | bold | size(WIDTH, EQUAL, 3),
            text("Now") | bold | size(WIDTH, EQUAL, 5),
            text("Edge ") | bold | size(WIDTH, EQUAL, 7),
            text("Type  ") | bold | size(WIDTH, EQUAL, 8),
            text("Legs") | bold | size(WIDTH, EQUAL, 16),
            text("Rationale") | bold | flex,
            text("Risk") | bold | size(WIDTH, EQUAL, 15),
        }));
        rows.push_back(separator());

        for (size_t i = 0; i < ideas.size(); ++i) {
            const auto& idea = ideas[i];

            std::string now_str = idea.tradable_now ? " ✓ " : " ✗ ";
            Color now_c = idea.tradable_now ? Color::Green : Color::Red;

            std::string legs_str;
            for (const auto& leg : idea.instruments) {
                if (!legs_str.empty()) legs_str += "/";
                legs_str += (leg.direction == "long" ? "+" : "-") + leg.symbol;
            }

            std::string risk_str;
            for (const auto& tag : idea.risk_tags) {
                if (!risk_str.empty()) risk_str += ",";
                risk_str += tag;
            }

            std::string rationale_short = idea.rationale.substr(0, 45);
            if (idea.rationale.size() > 45) rationale_short += "...";

            bool is_selected = (static_cast<int>(i) == selected);
            auto row_elem = hbox({
                text(std::to_string(i + 1)) | size(WIDTH, EQUAL, 3),
                text(now_str) | color(now_c) | size(WIDTH, EQUAL, 5),
                text(fmt_double(idea.expected_edge, 3)) | color(Color::Cyan) | size(WIDTH, EQUAL, 7),
                text(idea.trade_type) | size(WIDTH, EQUAL, 8),
                text(legs_str) | size(WIDTH, EQUAL, 16),
                text(rationale_short) | dim | flex,
                text(risk_str) | color(Color::Yellow) | size(WIDTH, EQUAL, 15),
            });

            if (is_selected) {
                row_elem = row_elem | inverted;
            }
            rows.push_back(row_elem);
        }
    }

    return vbox(rows) | border | flex;
}

static Element render_detail_modal(const AppState& state) {
    auto ideas = state.get_trade_ideas();
    int idx = state.get_selected_idea();

    if (idx < 0 || idx >= static_cast<int>(ideas.size())) {
        return text("No trade idea selected") | center | border;
    }

    const auto& idea = ideas[idx];

    Elements content;
    content.push_back(text(" TRADE IDEA DETAIL: " + idea.id) | bold | color(Color::Cyan));
    content.push_back(separator());

    content.push_back(hbox({text(" Type: ") | bold, text(idea.trade_type)}));
    content.push_back(hbox({text(" Edge: ") | bold, text(fmt_double(idea.expected_edge, 4)) | color(Color::Cyan)}));
    content.push_back(hbox({text(" Conf: ") | bold, text(fmt_double(idea.confidence, 3))}));
    content.push_back(hbox({
        text(" Tradable: ") | bold,
        text(idea.tradable_now ? "YES" : "NO") | color(idea.tradable_now ? Color::Green : Color::Red),
    }));

    if (!idea.best_window.empty()) {
        content.push_back(hbox({text(" Next window: ") | bold, text(idea.best_window)}));
    }

    content.push_back(separator());
    content.push_back(text(" INSTRUMENTS") | bold);
    for (const auto& leg : idea.instruments) {
        content.push_back(text("  " + leg.direction + " " + leg.symbol +
                               " (" + leg.instrument_name + ") w=" + fmt_double(leg.weight, 2)));
    }

    content.push_back(separator());
    content.push_back(text(" RATIONALE") | bold);
    // Word-wrap rationale
    content.push_back(paragraph(idea.rationale) | dim);

    content.push_back(separator());
    content.push_back(text(" HEDGES") | bold);
    for (const auto& h : idea.hedges) {
        content.push_back(text("  • " + h) | color(Color::GrayLight));
    }

    content.push_back(separator());
    content.push_back(text(" RISK NOTES") | bold);
    for (const auto& rn : idea.risk_notes) {
        content.push_back(text("  ! " + rn) | color(Color::Yellow));
    }

    content.push_back(separator());
    content.push_back(hbox({
        text(" Pin Risk: ") | bold,
        text(fmt_double(idea.pin_risk_score, 3)) | color(idea.pin_risk_score > 0.5 ? Color::Red : Color::Green),
    }));

    return vbox(content) | border | size(WIDTH, LESS_THAN, 80) | size(HEIGHT, LESS_THAN, 30);
}

static Element render_footer() {
    return hbox({
        text(" DISCLAIMER: For research only; not investment advice. No trades are auto-executed. ") | dim | color(Color::Yellow),
    }) | bgcolor(Color::GrayDark);
}

// ---------------------------------------------------------------------------
// Main UI builder
// ---------------------------------------------------------------------------

Component BuildUI(AppState& state, ScreenInteractive& screen) {
    auto renderer = Renderer([&state] {
        bool show_detail = state.get_show_detail();

        auto status_bar = render_status_bar(state);
        auto footer = render_footer();

        if (show_detail) {
            return vbox({
                status_bar,
                render_detail_modal(state) | center | flex,
                footer,
            });
        }

        int tab = state.get_active_tab();

        Element main_content;
        if (tab == 1) {
            main_content = render_macro_panel(state) | flex;
        } else if (tab == 2) {
            main_content = render_regions_panel(state) | flex;
        } else if (tab == 3) {
            main_content = render_trades_panel(state) | flex;
        } else {
            // Default: three columns
            main_content = hbox({
                render_macro_panel(state) | size(WIDTH, EQUAL, 40),
                render_regions_panel(state) | flex,
                render_trades_panel(state) | flex,
            });
        }

        return vbox({
            status_bar,
            main_content | flex,
            footer,
        });
    });

    // Add key handling
    auto component = CatchEvent(renderer, [&state, &screen](Event event) {
        if (event == Event::Character('q') || event == Event::Escape) {
            screen.Exit();
            return true;
        }
        if (event == Event::Character('r')) {
            // Reconnect is handled by ws_client auto-reconnect
            return true;
        }
        if (event == Event::Character('1')) {
            state.set_active_tab(1);
            return true;
        }
        if (event == Event::Character('2')) {
            state.set_active_tab(2);
            return true;
        }
        if (event == Event::Character('3')) {
            state.set_active_tab(3);
            return true;
        }
        if (event == Event::Character('0')) {
            state.set_active_tab(0);
            return true;
        }
        if (event == Event::Return) {
            if (state.get_show_detail()) {
                state.set_show_detail(false);
            } else {
                auto ideas = state.get_trade_ideas();
                if (!ideas.empty()) {
                    int sel = state.get_selected_idea();
                    if (sel < 0) sel = 0;
                    state.set_selected_idea(sel);
                    state.set_show_detail(true);
                }
            }
            return true;
        }
        if (event == Event::ArrowDown) {
            auto ideas = state.get_trade_ideas();
            int sel = state.get_selected_idea();
            if (sel < static_cast<int>(ideas.size()) - 1) {
                state.set_selected_idea(sel + 1);
            }
            return true;
        }
        if (event == Event::ArrowUp) {
            int sel = state.get_selected_idea();
            if (sel > 0) {
                state.set_selected_idea(sel - 1);
            }
            return true;
        }
        return false;
    });

    return component;
}

}  // namespace geoag
