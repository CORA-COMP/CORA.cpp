// What the libtorch backend of CoraTensor adds: batch dimensions, devices, backend specs,
// and the hand-written backward pass of the matrix exponential.

#include "tensor/eigen.h"
#include "tensor/tensor.h"
#include "tensor/torch.h"
#include "testing.h"

#include <stdexcept>

using namespace cora::ct;
using test::check;

namespace {

template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

void batches_broadcast() {
    set_backend("torch");
    const torch::Tensor batch = torch::randn({3, 2, 2}, torch::kDouble);
    const torch::Tensor single = torch::randn({2, 2}, torch::kDouble);
    const Tensor B = from_torch(batch), S = from_torch(single);

    check(B.matmul(S).shape() == std::vector<int64_t>({3, 2, 2}), "matmul broadcasts the batch");
    check((B + S).shape() == std::vector<int64_t>({3, 2, 2}), "add broadcasts the batch");
    check(max_diff(to_torch(B.matmul(S)), batch.matmul(single)) < 1e-14, "the batched product");
    check(B.transpose().shape() == std::vector<int64_t>({3, 2, 2}), "transpose keeps the batch");
    check(B.sum_last().shape() == std::vector<int64_t>({3, 2, 1}), "sum_last keeps the batch");
    check(B.eye_like().shape() == std::vector<int64_t>({3, 2, 2}), "eye_like takes the batch shape");
    check(Tensor::cat_last({B, B}).shape() == std::vector<int64_t>({3, 2, 4}), "cat_last per batch");
    check(from_torch(torch::randn({3, 2, 1}, torch::kDouble)).diag().shape() ==
              std::vector<int64_t>({3, 2, 2}),
          "diag per batch");

    // The exponential of a batch is the batch of exponentials.
    const torch::Tensor e = to_torch(B.expm());
    bool same = true;
    for (int i = 0; i < 3; ++i) same &= max_diff(e[i], torch::matrix_exp(batch[i])) < 1e-13;
    check(same, "expm per batch element");
}

void devices() {
    set_backend("torch");
    const Tensor t = Tensor::zeros({2, 2});
    check(t.device() == "cpu", "the default device is the CPU");
    check(Tensor::zeros({2, 2}, "cpu").device() == "cpu", "an explicit cpu");
    check(Tensor({1.0, 2.0}, "cpu").device() == "cpu", "a column on the cpu");
    check(throws([] { Tensor::zeros({2, 2}, "tpu"); }), "an unknown device");

    if (torch::cuda::is_available()) {
        check(Tensor::zeros({2, 2}, "gpu").device().rfind("cuda", 0) == 0, "zeros allocated on the gpu");
        check(Tensor({{1.0, 2.0}, {3.0, 4.0}}, "gpu").device().rfind("cuda", 0) == 0, "a matrix on the gpu");
        check(Tensor::eye(3, "cuda:0").device().rfind("cuda", 0) == 0, "eye on cuda:0");
        const Tensor g = t.to("gpu");
        check(g.device().rfind("cuda", 0) == 0 && g.to("cpu").device() == "cpu", "to() moves both ways");
        // Results stay where their operands are, and mixed devices are an error.
        check(g.matmul(g).device().rfind("cuda", 0) == 0, "an operation stays on the gpu");
        check(throws([&] { (void)(g + t); }), "cpu and gpu tensors do not mix");
        check(test::close(Tensor({{1.0, 2.0}, {3.0, 4.0}}, "gpu").matmul(Tensor::eye(2, "gpu")),
                          Tensor({{1.0, 2.0}, {3.0, 4.0}})),
              "values read back from the gpu");
        // A backend default on the gpu allocates there without asking.
        set_backend("torch:cuda");
        check(Tensor::zeros({2, 2}).device().rfind("cuda", 0) == 0, "torch:cuda is the default device");
        check(Tensor::zeros({2, 2}, "cpu").device() == "cpu", "and a tensor can still choose the cpu");
        set_backend("torch");
    } else {
        std::cout << "no CUDA device: the gpu half of these tests did not run\n";
    }

    set_backend("eigen");
    check(throws([] { Tensor::zeros({2, 2}, "gpu"); }), "eigen cannot allocate on the gpu");
    check(Tensor::zeros({2, 2}, "cpu").device() == "cpu", "eigen on the cpu");
    check(throws([] { Tensor::zeros({2, 2}).to("gpu"); }), "eigen cannot move to the gpu");
}

void backend_specs() {
    check(!throws([] { set_backend("torch:cpu,custom_backward"); }), "device and option together");
    check(!throws([] { set_backend("torch,custom_backward"); }), "an option alone");
    check(throws([] { set_backend("torch,nonsense"); }), "an unknown option");
    check(throws([] { set_backend("torch:tpu"); }), "an unknown device");
    set_backend("eigen");
    const Tensor on_eigen({1.0, 2.0});
    set_backend("torch");
    const Tensor on_torch({1.0, 2.0});
    check(throws([&] { (void)(on_eigen + on_torch); }), "tensors of two backends do not mix");
    check(throws([&] { on_eigen.matmul(on_torch.transpose()); }), "nor in a product");
    check(throws([&] { Tensor::cat_last({on_eigen, on_torch}); }), "nor in a concatenation");
}

/// The hand-written backward pass of `e^A` equals autograd's own.
void custom_backward_matches_autograd() {
    torch::manual_seed(1);
    for (const int64_t n : {1, 3, 5}) {
        const torch::Tensor A0 = torch::randn({2, n, n}, torch::kDouble) * 0.7;
        const torch::Tensor weight = torch::randn({2, n, n}, torch::kDouble);
        std::vector<torch::Tensor> grads;
        for (const bool custom : {false, true}) {
            torch::Tensor A = A0.clone().requires_grad_(true);
            const torch::Tensor loss = (to_torch(from_torch(A, custom).expm()) * weight).sum();
            grads.push_back(torch::autograd::grad({loss}, {A})[0]);
        }
        check(max_diff(grads[0], grads[1]) < 1e-10,
              "custom backward of expm, n = " + std::to_string(n));
    }
}

} // namespace

int main() {
    batches_broadcast();
    devices();
    backend_specs();
    custom_backward_matches_autograd();
    set_backend("eigen");
    return test::finish("tensor (libtorch)");
}
