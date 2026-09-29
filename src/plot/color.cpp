// color - CORA's colors, as MATLAB CORA's CORAcolor
//
// Syntax:   Color c = CORAcolor("CORA:reachSet", numColors, cidx);
// Inputs:   identifier - "CORA:<name>[:light|:dark]"; numColors, cidx - shading of "reachSet";
//           alpha - the mix of the ":light" and ":dark" variants
// Outputs:  c - the RGB color
// See also: plot/plot.h

#include "plot/color.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

std::vector<std::string> aux_split(const std::string &s) {
    std::vector<std::string> parts;
    std::stringstream in(s);
    for (std::string part; std::getline(in, part, ':');) parts.push_back(part);
    return parts;
}

// The default palette of MATLAB, which CORA uses as color1 .. color7.
const std::vector<std::pair<std::string, Color>> &aux_palette() {
    static const std::vector<std::pair<std::string, Color>> palette = {
        {"blue", {0.0, 0.4470, 0.7410}},       {"red", {0.8500, 0.3250, 0.0980}},
        {"yellow", {0.9290, 0.6940, 0.1250}},  {"purple", {0.4940, 0.1840, 0.5560}},
        {"green", {0.4660, 0.6740, 0.1880}},   {"light-blue", {0.3010, 0.7450, 0.9330}},
        {"dark-red", {0.6350, 0.0780, 0.1840}}};
    return palette;
}

Color aux_named(const std::string &name, const std::string &identifier, int numColors, int cidx) {
    static const std::map<std::string, Color> special = {
        {"initialSet", {1.0, 1.0, 1.0}},           {"finalSet", {0.9, 0.9, 0.9}},
        {"simulations", {0.0, 0.0, 0.0}},          {"unsafe", {0.9451, 0.5529, 0.5686}},
        {"unsafeLight", {0.9059, 0.7373, 0.7373}}, {"safe", {0.4706, 0.7725, 0.4980}},
        {"invariant", {0.4706, 0.7725, 0.4980}},   {"highlight1", {1.0, 0.6824, 0.2980}},
        {"highlight2", {0.6235, 0.7294, 0.2118}}};
    const Color main(0.2706, 0.5882, 1.0), worse(0.6902, 0.8235, 1.0);
    if (name == "reachSet") {
        if (cidx > numColors)
            throw std::invalid_argument("CORAcolor: the color index must not exceed the number "
                                        "of colors");
        if (cidx == numColors) return main;
        if (cidx == 1) return worse;
        // In between, the shade moves linearly from the light to the full blue.
        const double t = (cidx - 1.0) / (numColors - 1.0);
        return {worse.r + (main.r - worse.r) * t, worse.g + (main.g - worse.g) * t,
                worse.b + (main.b - worse.b) * t};
    }
    // The remaining names: "next", the special colors, and the palette by name or number.
    const auto &palette = aux_palette();
    if (name == "next") return palette[static_cast<std::size_t>((cidx - 1) % 7)].second;
    if (special.count(name)) return special.at(name);
    for (std::size_t i = 0; i < palette.size(); ++i)
        if (name == palette[i].first || name == "color" + std::to_string(i + 1))
            return palette[i].second;
    throw std::invalid_argument("CORAcolor: not a CORA color: '" + identifier +
                                "' is not a CORA color; see the "
                                "list in plot/color.h");
}

} // namespace


// ===========================================  MAIN  =========================================== //

Color::Color(const char *coraColor) : Color(CORAcolor(coraColor)) {}
Color::Color(const std::string &coraColor) : Color(CORAcolor(coraColor)) {}

// Each component as two hex digits.
std::string Color::hex() const {
    char out[8];
    auto byte = [](double v) { return static_cast<int>(v * 255 + 0.5); };
    std::snprintf(out, sizeof out, "#%02x%02x%02x", byte(r), byte(g), byte(b));
    return out;
}

// The identifier is "CORA:<name>[:variant]"; the variant mixes the named color with white or black.
Color CORAcolor(const std::string &identifier, int numColors, int cidx, double alpha) {
    const std::vector<std::string> parts = aux_split(identifier);
    if (parts.size() < 2 || parts[0] != "CORA")
        throw std::invalid_argument(
            "CORAcolor: not a CORA color: '" + identifier +
            "' is not a CORA color; use \"CORA:<name>\", e.g. \"CORA:blue\"");
    Color c = aux_named(parts[1], identifier, numColors, cidx);
    const std::string variant = parts.size() > 2 ? parts[2] : "none";
    auto mix = [&](double base) {
        c = Color(base * (1 - alpha) + c.r * alpha, base * (1 - alpha) + c.g * alpha,
                  base * (1 - alpha) + c.b * alpha);
    };
    if (variant == "light")
        mix(1.0);
    else if (variant == "dark")
        mix(0.0);
    else if (variant != "none")
        throw std::invalid_argument("CORAcolor: unknown color variant '" + variant +
                                    "'; use \":light\" or \":dark\"");
    return c;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
