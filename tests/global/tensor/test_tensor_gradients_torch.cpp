// test_tensor_gradients_torch - not differentiable operations detach and warn once, or fail
//
// On torch, an operation tagged not_differentiable that gets a tensor needing gradients computes
// on the detached tensor and warns once per operation; under GradientsScope it is an error.

#include "cora/global/backend/torch.h"
#include "cora/global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;

int main() {
    const BackendScope scope("torch");
    int warnings = 0;
    const WarningHandler previous =
        set_warning_handler([&](const std::string &id, const std::string &) {
            if (id == "CORA:notDifferentiable") ++warnings;
        });
    const Tensor x = from_torch(torch::randn({3, 4}, torch::kDouble).requires_grad_(true));
    check(x.requires_grad(), "a tensor that needs gradients");
    const Tensor order = x.argsort(-1);
    check(!order.requires_grad(), "argsort's result carries no gradient");
    (void)x.argsort(0);
    (void)x.argmax(-1);
    check(warnings == 2, "one warning per operation: " + std::to_string(warnings));
    check(x.sort(-1).requires_grad(), "sort, piecewise, keeps its gradient");
    {
        const GradientsScope strict;
        check(test::raises([&] { (void)x.argsort(-1); }, Error::not_supported),
              "strict gradients: an error");
        check(x.abs().requires_grad(), "a differentiable operation is fine when strict");
    }
    check(!x.argsort(-1).requires_grad(), "outside the scope, a warning again (already given)");
    check(warnings == 2, "and no second warning");
    std::size_t tagged = 0;
    for (const OpInfo &op : tensor_ops()) tagged += op.gradient == Gradient::not_differentiable;
    check(tagged > 0 && tensor_ops().size() == kOpCount, "the table lists every operation");
    set_warning_handler(previous);
    return test::finish("tensor gradients (libtorch)");
}
