// test_tensor_backend_scope_torch - backend specs for libtorch, and backends that do not mix

#include "cora/global/tensor/tensor.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

namespace {

void backend_specs() {
    check(!throws([] { set_default_backend("torch:cpu,custom_backward"); }), "device and option together");
    check(!throws([] { set_default_backend("torch,custom_backward"); }), "an option alone");
    check(throws([] { set_default_backend("torch,nonsense"); }), "an unknown option");
    check(throws([] { set_default_backend("torch:tpu"); }), "an unknown device");
    set_default_backend("eigen");
    const Tensor on_eigen({1.0, 2.0});
    set_default_backend("torch");
    const Tensor on_torch({1.0, 2.0});
    check(throws([&] { (void)(on_eigen + on_torch); }), "tensors of two backends do not mix");
    check(throws([&] { on_eigen.matmul(on_torch.transpose()); }), "nor in a product");
    check(throws([&] { Tensor::cat_last({on_eigen, on_torch}); }), "nor in a concatenation");
}

} // namespace

int main() {
    backend_specs();
    set_default_backend("eigen");
    return test::finish("tensor backend scope (libtorch)");
}
