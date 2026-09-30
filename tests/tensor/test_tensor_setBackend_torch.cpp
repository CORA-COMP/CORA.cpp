// test_tensor_setBackend_torch - setBackend specs for libtorch, and backends that do not mix

#include "tensor/tensor.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

namespace {

void backend_specs() {
    check(!throws([] { setBackend("torch:cpu,customBackward"); }), "device and option together");
    check(!throws([] { setBackend("torch,customBackward"); }), "an option alone");
    check(throws([] { setBackend("torch,nonsense"); }), "an unknown option");
    check(throws([] { setBackend("torch:tpu"); }), "an unknown device");
    setBackend("eigen");
    const Tensor on_eigen({1.0, 2.0});
    setBackend("torch");
    const Tensor on_torch({1.0, 2.0});
    check(throws([&] { (void)(on_eigen + on_torch); }), "tensors of two backends do not mix");
    check(throws([&] { on_eigen.matmul(on_torch.transpose()); }), "nor in a product");
    check(throws([&] { Tensor::catLast({on_eigen, on_torch}); }), "nor in a concatenation");
}

} // namespace

int main() {
    backend_specs();
    setBackend("eigen");
    return test::finish("tensor setBackend (libtorch)");
}
