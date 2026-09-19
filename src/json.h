// Just enough JSON for the catalog's `params`.
//
// Every instance is a flat object of strings and numbers, so a scanner for one key at a
// time is all this needs — and it keeps the tool to Eigen and GLPK.

#pragma once

#include <optional>
#include <string>

namespace cora {

/// The string value of `key`, or nothing if it is absent or not a string.
std::optional<std::string> json_string(const std::string &doc, const std::string &key);

/// The numeric value of `key`, or nothing if it is absent or not a number.
std::optional<double> json_number(const std::string &doc, const std::string &key);

} // namespace cora
