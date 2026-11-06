// options - the field lists of option structs, from which Python gets its keyword arguments
//
// A problem (CORA's params) or an algorithm (CORA's options) is a plain struct with defaults.
// Its static fields() lists every member once, by name, so the Python bindings build the
// struct from keyword arguments, and an unknown keyword is refused with the closest name.
//
// Syntax:   static constexpr auto fields() { return std::make_tuple(field("time_step",
// &S::time_step)); }
//           std::string hint = did_you_mean("time_stp", {"time_step", "t_final"});
// See also: cont_dynamics/cont_dynamics.h, cont_dynamics/linear_sys/linear_sys.h
//
// CORA.cpp only: MATLAB CORA checks its structs at run time against a table (validateOptions).
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <string>
#include <tuple>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// One member of an option struct: its name and where it is.
template <class S, class T>
struct Field {
    const char *name;
    T S::*member;
};

/// The field `name` of `member`.
template <class S, class T>
constexpr Field<S, T> field(const char *name, T S::*member) {
    return {name, member};
}

/// The names of the fields of S.
template <class S>
std::vector<std::string> field_names() {
    std::vector<std::string> names;
    std::apply([&](const auto &...f) { (names.push_back(f.name), ...); }, S::fields());
    return names;
}

/// " (did you mean 'x'?)" for the candidate closest to `name` by edit distance, if one is close
/// enough, else "".
std::string did_you_mean(const std::string &name, const std::vector<std::string> &candidates);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
