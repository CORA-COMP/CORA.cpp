// The libtorch backend: its operations against their definitions, its containment against
// the Eigen backend's, and its gradients against the analytic ones.
//
// The cross-check is the point of this file. Each backend already agrees with the LP on
// its own, which catches an inconsistency inside one of them but not a shared misreading
// of the catalog; running the *same* sets through both and comparing the answers does.
// Every test runs on the CPU and, when the worker has one, on the GPU.
//
// Built and run by `make test` whenever the tool was built with TORCH=...; without it
// this file is not compiled at all.

#include "backend.h"
#include "probes.h"
#include "sets.h"
#include "torch_sets.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

/// The two set types have the same name in either backend, so the test names both.
using TensorZonotope = cora::tb::Zonotope;
using TensorInterval = cora::tb::Interval;
using EigenZonotope = cora::Zonotope<double>;

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

/// An Eigen matrix as a tensor of the same shape, on `opts`' device.
torch::Tensor to_tensor(const Eigen::Ref<const Eigen::MatrixXd> &m,
                        const torch::TensorOptions &opts) {
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rows = m;
    return torch::from_blob(const_cast<double *>(rows.data()), {m.rows(), m.cols()},
                            torch::TensorOptions().dtype(torch::kDouble))
        .clone()
        .to(opts);
}

void the_operations_match_their_definitions(const torch::TensorOptions &opts,
                                            const std::string &where) {
    const int64_t n = 3, m = 5, batch = 2;
    const TensorZonotope zono = TensorZonotope::generate_random({batch}, n, m, opts);
    const TensorInterval box = TensorInterval::generate_random({batch}, n, opts);
    const torch::Tensor d = torch::randn({batch, n}, opts);
    const torch::Tensor matrix = torch::randn({n, n}, opts);

    // A zonotope's support is c'd + ||G'd||_1.
    const torch::Tensor want =
        (zono.c * d).sum(-1) + d.unsqueeze(-2).matmul(zono.g).abs().sum({-2, -1});
    check((zono.support_func(d) - want).abs().max().item<double>() < 1e-12,
          where + ": the zonotope support function");

    // M maps every generator alike.
    check((zono.mat_mul(matrix).g - matrix.matmul(zono.g)).abs().max().item<double>() < 1e-12,
          where + ": matMul maps the generators");

    // Every generator was scaled to a length in [0, 1].
    check(zono.g.norm(2, {-2}, false).max().item<double>() <= 1.0 + 1e-12,
          where + ": the generator lengths");

    check(box.contains(box.rand_point(20)).all().item<bool>(),
          where + ": an interval contains the points drawn from it");
    check(zono.mink_sum(zono).g.size(-1) == 2 * m, where + ": minkSum concatenates");
}

/// The same sets and the same points through both backends.
void the_two_backends_agree(const torch::TensorOptions &opts, const std::string &where) {
    cora::Rng rng(11);
    struct Shape {
        Eigen::Index n, m;
    };
    // 5x2500 is past the projector the tensor path is willing to hold as a matrix, so it
    // takes the factored form that the catalog's largest instances run on; every smaller
    // shape here takes the held one. Both paths are checked against the same LP.
    const Shape shapes[] = {{1, 2}, {2, 4}, {3, 6}, {5, 10}, {10, 20}, {30, 60}, {5, 2500}};
    for (const auto [n, m] : shapes) {
        const Eigen::Index batch = 3, points = 12;
        const EigenZonotope set = cora::random_zonotope(rng, n, m, batch);
        const cora::Mat<double> probes = cora::across_the_boundary(set, points, rng);

        // Eigen keeps one set per column with the generator blocks side by side; a tensor
        // wants c as (B, n), G as (B, n, m) and the points as (B, N, n).
        const torch::Tensor c = to_tensor(set.c, opts).transpose(0, 1).contiguous();
        const torch::Tensor g =
            to_tensor(set.g, opts).reshape({n, batch, m}).permute({1, 0, 2}).contiguous();
        const torch::Tensor p =
            to_tensor(probes, opts).reshape({n, batch, points}).permute({1, 2, 0}).contiguous();

        const cora::Mask from_eigen = cora::contains(set, probes, points);
        const torch::Tensor from_torch =
            TensorZonotope{c, g}.contains(p).to(torch::kCPU).reshape({-1}).contiguous();
        const bool *answers = from_torch.data_ptr<bool>();

        bool same = true;
        for (std::size_t k = 0; k < from_eigen.size(); ++k)
            same &= (from_eigen[k] != 0) == answers[k];
        check(same,
              where + ": the backends disagreed about containment at " + std::to_string(n)
                  + "x" + std::to_string(m));
    }
}

} // namespace

int main() {
    std::vector<std::string> devices{"cpu"};
    if (torch::cuda::is_available()) devices.emplace_back("gpu");

    for (const std::string &device : devices) {
        const auto opts = torch::TensorOptions().dtype(torch::kDouble).device(
            device == "gpu" ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU));
        torch::manual_seed(7);
        the_operations_match_their_definitions(opts, device);
        the_two_backends_agree(opts, device);
        cora::torch_backend::check_gradients(device); // throws when they are wrong
    }
    if (devices.size() == 1)
        std::cout << "no CUDA device: the gpu half of these tests did not run\n";
    if (failures == 0) std::cout << "all libtorch tests passed\n";
    return failures == 0 ? 0 : 1;
}
