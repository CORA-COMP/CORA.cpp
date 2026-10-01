// format - numbers and matrices as text, for the display of sets
//
// Syntax:   std::string s = formatMatrix(t, "    ");
// See also: contSet/zonotope/display.cpp, contSet/interval/display.cpp

#pragma once

#include "global/tensor/tensor.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// One number with up to six significant digits.
inline std::string formatNumber(double x) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.6g", x);
    return buffer;
}

/// The matrix (rows, cols) row by row, the numbers right-aligned in columns and every row
/// starting with `indent`. A tensor with batch dimensions is described by its shape instead.
inline std::string formatMatrix(const Tensor &t, const std::string &indent) {
    const std::vector<int64_t> shape = t.shape();
    if (shape.size() != 2) {
        std::string text = indent + "batch of shape (";
        for (std::size_t i = 0; i < shape.size(); ++i)
            text += (i ? ", " : "") + std::to_string(shape[i]);
        return text + ")\n";
    }
    const std::vector<double> values = t.data();
    std::size_t width = 0;
    for (const double v : values) width = std::max(width, formatNumber(v).size());
    std::string text;
    for (int64_t i = 0; i < shape[0]; ++i) {
        text += indent;
        for (int64_t j = 0; j < shape[1]; ++j) {
            const std::size_t at = static_cast<std::size_t>(i * shape[1] + j);
            const std::string cell = formatNumber(values[at]);
            text += std::string(width - cell.size(), ' ') + cell + (j + 1 < shape[1] ? "  " : "");
        }
        text += "\n";
    }
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
