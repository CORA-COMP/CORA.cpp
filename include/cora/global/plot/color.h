// color - CORA's colors, as MATLAB CORA's CORAcolor
//
// Syntax:   Color c = cora_color("CORA:reachSet");   Color c = "CORA:red";   Color c(0.2, 0.4, 0.8);
// See also: cora/global/plot/plot.h

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

/// The color of `identifier`: "CORA:reachSet", "CORA:initialSet", "CORA:finalSet",
/// "CORA:simulations", "CORA:unsafe", "CORA:unsafeLight", "CORA:safe", "CORA:invariant",
/// "CORA:highlight1", "CORA:highlight2", "CORA:next", a palette name ("CORA:blue", "CORA:red",
/// "CORA:yellow", "CORA:purple", "CORA:green", "CORA:light-blue", "CORA:dark-red") or its number
/// ("CORA:color3"). A postfix ":light" or ":dark" mixes it with white or black by `alpha`.
/// "CORA:reachSet" shades from light to full blue over `num_colors` sets, `cidx` of them (1 is
/// the lightest); "CORA:next" is the `cidx`-th color of the palette.
Color cora_color(const std::string &identifier, int num_colors = 1, int cidx = 1, double alpha = 0.2);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
