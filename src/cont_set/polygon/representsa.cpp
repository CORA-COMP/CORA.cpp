// representsa - whether a polygon is also a set of another type, as CORA's representsa_
//
// emptySet: no piece has a vertex (CORALean Polygon.representsaEmptySet); polygon: always.
// The other types are not ported.
//
// Syntax:   bool b = pgon.representsa("emptySet");
// Inputs:   type - "emptySet" or "polygon"
//           tol - not used
// Outputs:  b - whether the polygon is of that type
// See also: empty
//
// CORA: contSet/@polygon/representsa_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Polygon::representsa(const std::string &type, Tol) const {
    detail::Entry entry;
    if (type == "polygon") return true;
    if (type == "emptySet") {
        for (const Polytope &P : pieces_)
            if (P.V().size(0) > 0) return false;
        return true;
    }
    CORA_FAIL(Error::not_supported,
              "representsa: a polygon answers 'emptySet' and 'polygon' only, got '" + type + "'");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
