// lean - the lean backend of Tensor: double matrices whose operations round and order as CORALean's
//
// Syntax:   setBackend("lean");   Tensor t({{1, 2}, {3, 4}});   bool b = isLean(t);
// See also: tensor.h, eigen.h, lean/oracle.h

#pragma once

#include "global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The lean backend, for `setBackend`.
std::shared_ptr<const Tensor::Backend> leanBackend();

/// Whether the tensor belongs to the lean backend.
bool isLean(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
