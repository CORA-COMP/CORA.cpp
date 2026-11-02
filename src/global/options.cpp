// options - the closest name to a mistyped option, for the messages of the checks
//
// Syntax:   std::string hint = did_you_mean("time_stp", {"time_step", "t_final"});
// Inputs:   name - what was given;  candidates - the names there are
// Outputs:  hint - " (did you mean 'time_step'?)", or "" if no candidate is close
// See also: cora/global/options.h

#include "cora/global/options.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The Levenshtein distance of a and b.
std::size_t aux_distance(const std::string &a, const std::string &b) {
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        std::size_t diagonal = row[0];
        row[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] != b[j - 1])});
            diagonal = above;
        }
    }
    return row[b.size()];
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::string did_you_mean(const std::string &name, const std::vector<std::string> &candidates) {
    std::string best;
    std::size_t best_distance = 0;
    for (const std::string &c : candidates) {
        const std::size_t d = aux_distance(name, c);
        if (best.empty() || d < best_distance) best = c, best_distance = d;
    }
    // close: at most a third of the longer name differs
    if (best.empty() || 3 * best_distance > std::max(name.size(), best.size())) return "";
    return " (did you mean '" + best + "'?)";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
