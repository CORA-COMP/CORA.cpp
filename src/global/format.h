// format - numbers, shapes and matrices as text, for the display of sets and for the checks
//
// Syntax:   std::string s = format_matrix(t, "    ");
// See also: src/cont_set/zonotope/display.cpp, src/cont_set/interval/display.cpp
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       29-September-2026
// Last update:   02-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/error.h"
#include "cora/global/tensor/tensor.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A shape as text, "(2, 3)", for the messages of the checks.
inline std::string shape_text(const std::vector<int64_t> &shape) {
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i)
        text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + ")";
}

/// Whether t is a column (..., n, 1) of n rows.
inline bool is_column(const Tensor &t, int64_t n) {
    const std::vector<int64_t> shape = t.shape();
    return shape.size() >= 2 && shape.back() == 1 && shape[shape.size() - 2] == n;
}

/// One number with up to six significant digits.
inline std::string format_number(double x) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.6g", x);
    return buffer;
}

/// Writes `text` to the file `path`; `who` names the caller in the error.
inline void write_text_file(const std::string &path, const std::string &text,
                            const std::string &who) {
    std::ofstream file(path);
    if (!file) CORA_FAIL(Error::special_error, who + ": cannot write '" + path + "'");
    file << text;
}

/// The matrix (rows, cols) row by row, the numbers right-aligned in columns and every row
/// starting with `indent`. A tensor with batch dimensions is described by its shape instead.
inline std::string format_matrix(const Tensor &t, const std::string &indent) {
    const std::vector<int64_t> shape = t.shape();
    if (shape.size() != 2) {
        std::string text = indent + "batch of shape (";
        for (std::size_t i = 0; i < shape.size(); ++i)
            text += (i ? ", " : "") + std::to_string(shape[i]);
        return text + ")\n";
    }
    const std::vector<double> values = t.data();
    std::size_t width = 0;
    for (const double v : values) width = std::max(width, format_number(v).size());
    std::string text;
    for (int64_t i = 0; i < shape[0]; ++i) {
        text += indent;
        for (int64_t j = 0; j < shape[1]; ++j) {
            const std::size_t at = static_cast<std::size_t>(i * shape[1] + j);
            const std::string cell = format_number(values[at]);
            text += std::string(width - cell.size(), ' ') + cell + (j + 1 < shape[1] ? "  " : "");
        }
        text += "\n";
    }
    return text;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
