// host - the CPU memory of eigen and lean tensors, shared without a copy
//
// A host tensor can borrow memory it does not own (a numpy array): it keeps the owner alive and
// never writes to it. Its own memory can be lent out the same way (a numpy array over a result).
//
// Syntax:   Tensor t = host_borrow(ptr, DType::float64, {n, m}, owner, backend_id::eigen);
//           HostView v = host_view(t);   // v.data, v.dtype, v.shape, v.keep
// See also: eigen.h, lean.h, src/global/python/convert.h

#pragma once

#include "cora/global/tensor/tensor.h"

#include <memory>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether the tensor lives on the host backends (eigen or lean).
bool is_host(const Tensor &t);

/// A host tensor of the backend `backend` (backend_id::eigen or lean) over row-major `data`;
/// `owner` keeps the memory alive. No copy.
Tensor host_borrow(const void *data, DType dtype, const Shape &shape,
                   const std::shared_ptr<const void> &owner, uint8_t backend);

/// The memory of a host tensor: row-major entries, and what keeps them alive.
struct HostView {
    const void *data;
    DType dtype;
    Shape shape;
    std::shared_ptr<const void> keep;
    /// Whether nobody else holds the memory, so a receiver may write to it.
    bool unique;
};
HostView host_view(const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
