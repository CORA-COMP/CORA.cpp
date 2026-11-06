// color - CORA's colors, as MATLAB CORA's CORAcolor
//
// Syntax:   Color c = cora_color("CORA:reachSet");   Color c = "CORA:red";   Color c(0.2, 0.4,
// 0.8); See also: cora/global/plot/plot.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// An RGB color, each component in [0, 1]. A CORA identifier converts to it implicitly, so
/// options read `{.color = "CORA:red"}`.
struct Color {
    double r = 0, g = 0, b = 0;

    Color() = default;
    Color(double r, double g, double b) : r(r), g(g), b(b) {}
    Color(const char *identifier);
    Color(const std::string &identifier);

    /// The color as "#rrggbb".
    std::string hex() const;
};

/// The color of `identifier`: a CORA name ("CORA:reachSet", "CORA:next", ...), a palette name
/// ("CORA:blue") or its number ("CORA:color3"); ":light" or ":dark" mixes it by `alpha`.
Color cora_color(const std::string &identifier, int num_colors = 1, int cidx = 1,
                 double alpha = 0.2);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
