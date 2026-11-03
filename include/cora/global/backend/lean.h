// lean - the lean backend of Tensor: float64 tensors whose operations round and order as CORALean's
//
// Syntax:   BackendScope scope("lean");   Tensor t({{1, 2}, {3, 4}});   bool b = is_lean(t);
// See also: tensor.h, eigen.h, lean/oracle.h

#pragma once

#include "cora/global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The lean backend, for `backend(spec)` and `BackendScope`.
std::shared_ptr<const Tensor::Backend> lean_backend();

/// Whether the tensor belongs to the lean backend.
bool is_lean(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
