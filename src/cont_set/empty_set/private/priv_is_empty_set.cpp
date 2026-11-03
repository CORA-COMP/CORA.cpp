// priv_is_empty_set - whether a set of any class has no point
//
// See also: priv_is_empty_set.h

#include "cont_set/empty_set/private/priv_is_empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_empty_set(const ContSet &S) { return S.center().shape().back() == 0; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
