// test_tensor_device_torch - Tensor devices: allocation on cpu and gpu, moving, and what does not
// mix

#include "global/tensor/tensor.h"
#include "global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

namespace {

void devices() {
    setBackend("torch");
    const Tensor t = Tensor::zeros({2, 2});
    check(t.device() == "cpu", "the default device is the CPU");
    check(Tensor::zeros({2, 2}, "cpu").device() == "cpu", "an explicit cpu");
    check(Tensor({1.0, 2.0}, "cpu").device() == "cpu", "a column on the cpu");
    check(throws([] { Tensor::zeros({2, 2}, "tpu"); }), "an unknown device");

    if (torch::cuda::is_available()) {
        check(Tensor::zeros({2, 2}, "gpu").device().rfind("cuda", 0) == 0,
              "zeros allocated on the gpu");
        check(Tensor({{1.0, 2.0}, {3.0, 4.0}}, "gpu").device().rfind("cuda", 0) == 0,
              "a matrix on the gpu");
        check(Tensor::eye(3, "cuda:0").device().rfind("cuda", 0) == 0, "eye on cuda:0");
        const Tensor g = t.to("gpu");
        check(g.device().rfind("cuda", 0) == 0 && g.to("cpu").device() == "cpu",
              "to() moves both ways");
        // Results stay where their operands are, and mixed devices are an error.
        check(g.matmul(g).device().rfind("cuda", 0) == 0, "an operation stays on the gpu");
        check(throws([&] { (void)(g + t); }), "cpu and gpu tensors do not mix");
        check(test::close(Tensor({{1.0, 2.0}, {3.0, 4.0}}, "gpu").matmul(Tensor::eye(2, "gpu")),
                          Tensor({{1.0, 2.0}, {3.0, 4.0}})),
              "values read back from the gpu");
        // A backend default on the gpu allocates there without asking.
        setBackend("torch:cuda");
        check(Tensor::zeros({2, 2}).device().rfind("cuda", 0) == 0,
              "torch:cuda is the default device");
        check(Tensor::zeros({2, 2}, "cpu").device() == "cpu",
              "and a tensor can still choose the cpu");
        setBackend("torch");
    } else {
        std::cout << "no CUDA device: the gpu half of these tests did not run\n";
    }

    setBackend("eigen");
    check(throws([] { Tensor::zeros({2, 2}, "gpu"); }), "eigen cannot allocate on the gpu");
    check(Tensor::zeros({2, 2}, "cpu").device() == "cpu", "eigen on the cpu");
    check(throws([] { Tensor::zeros({2, 2}).to("gpu"); }), "eigen cannot move to the gpu");
}

} // namespace

int main() {
    devices();
    setBackend("eigen");
    return test::finish("tensor device (libtorch)");
}
