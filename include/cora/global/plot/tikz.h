// tikz - what the TikZ export of a figure offers
//
// Syntax:   figure().save_tikz("reach.tikz", {.standalone = true});
// See also: cora/global/plot/figure.h, experiments/paper/latex/cora-macs.sty (CORA's colors)

#pragma once


#include "cora/global/tensor/tolerance.h"
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// How a figure is written as pgfplots code. The text is the same for every call: numbers have a
/// fixed number of decimals, layers keep their order, and nothing depends on the machine.
struct TikzOptions {
    /// A complete LaTeX document (article class, pgfplots, the colors and styles it uses), so that
    /// the file compiles alone; else a tikzpicture to \input into a document.
    bool standalone = false;

    /// Colors of CORA by their names in cora-macs (CORAcolorReachSet, ...) and its `plot filled`
    /// style; a standalone file defines them itself. Without it, colors are written as RGB values.
    bool coramacs = true;

    /// Whether to remove vertices that lie within `thin_tolerance` of the line through their
    /// neighbors, as matlab2tikz's cleanfigure does; the tolerance is a fraction of the axis range.
    bool thin = true;
    double thin_tolerance = numerics::kThinTolerance;

    /// Whether the coordinates of every layer go to a table next to the .tikz file,
    /// `<path without extension>_<k>.dat`, which the .tikz file reads (by the path given to
    /// save_tikz, so compile from where it is valid). Only save_tikz can write tables.
    bool external_data = false;

    /// The size of each axes: a TeX length, e.g. "6cm" or "0.9\\linewidth"; empty is pgfplots's
    /// (in a group plot 5cm by 4cm).
    std::string width, height;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
