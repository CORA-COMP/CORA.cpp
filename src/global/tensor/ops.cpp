// ops - the table of Tensor operations (ops.def) and the gradient policy
//
// A not_differentiable operation given a tensor that needs gradients computes on the detached
// tensor and warns once per operation ("CORA:notDifferentiable"); under GradientsScope it is an
// error instead.
//
// Syntax:   for (const OpInfo &op : tensor_ops()) ...;   GradientsScope strict;
// See also: ops.def, tensor.h, docs/tensor.md
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/tensor/tensor.h"
#include "global/tensor/derived.h"

#include <atomic>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of ops.def, in their order (the order of TensorOp).
const std::vector<OpInfo> &aux_table() {
    static const std::vector<OpInfo> table = {
#define CORA_TENSOR_OP(name, kind, gradient, agreement, k, summary)                                \
    {#name, OpKind::kind, Gradient::gradient, Agreement::agreement, k, summary},
#include "cora/global/tensor/ops.def"
#undef CORA_TENSOR_OP
    };
    return table;
}

/// Whether the warning of each operation was given.
std::atomic<bool> aux_warned[kOpCount];

thread_local bool aux_strict = false;

} // namespace


// ===========================================  MAIN  =========================================== //

const std::vector<OpInfo> &tensor_ops() { return aux_table(); }

const OpInfo &op_info(TensorOp op) { return aux_table()[static_cast<std::size_t>(op)]; }

GradientsScope::GradientsScope(bool strict) : previous_(aux_strict) { aux_strict = strict; }

GradientsScope::~GradientsScope() { aux_strict = previous_; }

// One warning per operation and process; strict, an error.
Tensor detail::without_gradient(TensorOp op, const Tensor &t) {
    if (!t.requires_grad()) return t;
    const OpInfo &info = op_info(op);
    const std::string message = std::string("CoraTensor::") + info.name +
                                " is not differentiable: its result carries no gradient";
    if (aux_strict) CORA_FAIL(Error::not_supported, message + " (strict_gradients is on)");
    if (!aux_warned[static_cast<std::size_t>(op)].exchange(true))
        warn("CORA:notDifferentiable", message);
    return t.detach();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
