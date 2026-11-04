// compact - a set without redundancy, as CORA's contSet.compact and compact_
//
// Syntax:   R = S.compact();   R = S.compact("all", tol);
// Inputs:   S - a set;  method - a variant of the class (default: the class's own; see its
//           compact);  tol - nonnegative, default: the repository tolerance
// Outputs:  R - the set with the redundant parts removed, in the class of S; a box, the empty
//           set and the whole space are returned as they are, Error::noops for a class without
//           compact_
// See also: Zonotope::compact, reduce
//
// CORA: contSet/@contSet/compact.m
// CORA: contSet/@contSet/compact_.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::compact(const std::optional<std::string> &method, Tol tol) const {
    detail::Entry entry;
    if (const auto methods = compact_methods_impl()) {
        resolve_tol(entry, tol, "compact"); // the check of tol
        const bool known =
            !method || std::find(methods->begin(), methods->end(), *method) != methods->end();
        CORA_CHECK(known, Error::wrong_value, [&] {
            return "compact: the method '" + *method + "' is not one of the class's";
        });
    }
    return compact_impl(method, tol);
}

std::optional<std::vector<std::string>> ContSet::compact_methods_impl() const {
    return std::nullopt;
}

Set ContSet::compact_impl(const std::optional<std::string> &, Tol) const {
    priv_noops("compact", *this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
