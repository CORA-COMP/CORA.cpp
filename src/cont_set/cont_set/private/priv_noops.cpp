// priv_noops - the failure of an operation that a set class does not implement
//
// See also: priv_noops.h

#include "cont_set/cont_set/private/priv_noops.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void priv_noops(const char *operation, const ContSet &S) {
    CORA_FAIL(Error::noops, std::string(operation) + " is not implemented for " + S.class_name());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
