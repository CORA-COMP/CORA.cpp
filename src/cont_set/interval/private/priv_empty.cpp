// priv_empty - the empty interval of the shape of a tensor
//
// See also: priv_empty.h

#include "cont_set/interval/private/priv_empty.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_empty(const Tensor &like) {
    std::vector<int64_t> shape = like.shape();
    shape.back() = 0;
    return {like.zeros_like(shape), like.zeros_like(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
