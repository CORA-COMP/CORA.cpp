// priv_noops - the failure of an operation that a set class does not implement
//
// See also: priv_noops.h
//
// CORA.cpp only: CORA raises these from the abstract methods of contSet.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void priv_noops(const char *operation, const ContSet &S, const ContSet *other) {
    CORA_FAIL(Error::noops, std::string(operation) + " is not implemented for " + S.class_name() +
                                (other ? " and " + other->class_name() : std::string()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
