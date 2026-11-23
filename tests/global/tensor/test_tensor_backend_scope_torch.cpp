// test_tensor_backend_scope_torch - backend specs for libtorch, and backends that do not mix
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/backend/torch.h"
#include "cora/global/tensor/tensor.h"
#include "testing.h"

#include <cmath>
#include <vector>

using namespace cora;
using test::check;
using test::close;
using test::throws;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void backend_specs() {
    check(!throws([] { set_default_backend("torch:cpu,custom_backward"); }),
          "device and option together");
    check(!throws([] { set_default_backend("torch,custom_backward"); }), "an option alone");
    check(throws([] { set_default_backend("torch,nonsense"); }), "an unknown option");
    check(throws([] { set_default_backend("torch,tf32,nonsense"); }), "an unknown second option");
    // TensorFloat-32: allowed by ",tf32", off for every other torch backend made
    set_default_backend("torch,custom_backward,tf32");
    check(torch_tf32(), "tf32 with another option allows TensorFloat-32");
    set_default_backend("torch:cpu");
    check(!torch_tf32(), "a torch backend without tf32 turns it off");
    check(throws([] { set_default_backend("torch:tpu"); }), "an unknown device");
    set_default_backend("eigen");
    const Tensor on_eigen({{1.0}, {2.0}});
    set_default_backend("torch");
    const Tensor on_torch({{1.0}, {2.0}});
    check(throws([&] { (void)(on_eigen + on_torch); }), "tensors of two backends do not mix");
    check(throws([&] { on_eigen.matmul(on_torch.matrix_transpose()); }), "nor in a product");
    check(throws([&] { Tensor::concat({on_eigen, on_torch}, -1); }), "nor in a concatenation");
}

/// float32 products on CUDA in full precision: TensorFloat-32 would be about 1e-3 off.
void float32_products_on_cuda() {
    if (!torch::cuda::is_available()) return;
    const BackendScope scope("torch:cuda");
    std::vector<double> a(256 * 256), b(256 * 256);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = std::sin(double(i) * 0.37);
        b[i] = std::cos(double(i) * 0.61);
    }
    const Tensor A = Tensor::from_data(a, {256, 256}), B = Tensor::from_data(b, {256, 256});
    const Tensor exact = A.matmul(B);
    const Tensor single = A.to(DType::float32).matmul(B.to(DType::float32)).to(DType::float64);
    const double gap = (single - exact).abs().max().data()[0];
    const double scale = exact.abs().max().data()[0];
    check(gap <= 1e-5 * scale, "float32 products on CUDA without TensorFloat-32");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    backend_specs();
    float32_products_on_cuda();
    set_default_backend("eigen");
    return test::finish("tensor backend scope (libtorch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
