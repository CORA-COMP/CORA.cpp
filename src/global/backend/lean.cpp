// lean - the lean backend of Tensor: float64 tensors whose operations round and order as CORALean's
//
// The backend runs an algorithm of CORA.cpp with the arithmetic that the CORALean oracle uses at
// dtype "nearest": IEEE binary64, round-to-nearest, no fused multiply-add, and the sums of
// CORALean's `sumUp`, which nests to the right, `f 0 + (f 1 + (... + (f (n-1) + 0)))`. Its tensors
// are the reference HostTensors (host_tensor.h) with every operation as defined there, the derived
// and fused ones as their portable defaults: the result of an operation then equals the oracle's
// bit for bit. It is a reference, not a fast backend.

#include "cora/global/backend/lean.h"
#include "cora/global/error.h"
#include "global/backend/host_tensor.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Makes lean tensors: float64 on the CPU only.
struct LeanBackend : Tensor::Backend {
    std::string name() const override { return "lean"; }
    Tensor::Ptr make(const std::vector<double> &data, const Shape &shape, const std::string &device,
                     DType dtype) const override {
        if (!device.empty() && device != "cpu")
            CORA_FAIL(Error::wrong_value, "CoraTensor: the lean backend runs on the CPU only");
        if (dtype != DType::float64)
            CORA_FAIL(Error::not_supported,
                      "CoraTensor: the lean backend computes in float64 only");
        auto t = host::HostTensor<double>::make(shape, backend_id::lean);
        std::copy(data.begin(), data.end(), t->p);
        return t;
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Backend -----------------------------------------------------------------------------------------

std::shared_ptr<const Tensor::Backend> lean_backend() { return std::make_shared<LeanBackend>(); }

bool is_lean(const Tensor &t) { return t.impl().backend == backend_id::lean; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
