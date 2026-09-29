// `linearSys` reachability on libtorch: what the backend adds to the algorithms — batching
// over sets and over systems, gradients, the GPU — and that it agrees with Eigen on the
// same input, since the algorithms are written once and only the backend differs.
//
// Built and run by `make test` whenever the tool was built with TORCH=...

#include "contDynamics/linear_sys.h"
#include "tensor/eigen.h"
#include "tensor/torch.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cora::ct;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

std::string name(Algorithm a) { return a == Algorithm::Standard ? "standard" : "wrapping-free"; }

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// A set from a centre `(..., n)` and generators `(..., n, m)`.
Zonotope make_set(const torch::Tensor &c, const torch::Tensor &G, bool custom = false) {
    return {from_torch(c.unsqueeze(-1), custom), from_torch(G, custom)};
}

Eigen::MatrixXd to_matrix(const torch::Tensor &t) {
    const torch::Tensor rows = t.to(torch::kCPU).contiguous();
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
        rows.data_ptr<double>(), rows.size(0), rows.size(1));
}

/// The sets of a run flattened to one tensor, so runs can be compared in one go.
torch::Tensor flatten(const Reach &r) {
    std::vector<torch::Tensor> all;
    for (const auto *sets : {&r.time_int, &r.time_point})
        for (const Zonotope &Z : *sets) {
            all.push_back(to_torch(Z.c).reshape({-1}));
            all.push_back(to_torch(Z.G).reshape({-1}));
        }
    return torch::cat(all);
}

void the_backends_agree(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 4, m = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const torch::Tensor c = torch::randn({n}, opts), G = torch::randn({n, m}, opts);

    for (const Algorithm algorithm : kAlgorithms) {
        const Reach with_torch =
            LinearSys(from_torch(A)).reach(make_set(c, G), 0.1, 1.0, 8, algorithm);
        const Zonotope X0_eigen{from_eigen(to_matrix(c.unsqueeze(-1))), from_eigen(to_matrix(G))};
        const Reach with_eigen =
            LinearSys(from_eigen(to_matrix(A))).reach(X0_eigen, 0.1, 1.0, 8, algorithm);

        // Eigen's matrices are the same numbers in another container.
        std::vector<torch::Tensor> eigen_flat;
        for (const auto *sets : {&with_eigen.time_int, &with_eigen.time_point})
            for (const Zonotope &Z : *sets) {
                for (const Tensor *t : {&Z.c, &Z.G}) {
                    const Eigen::MatrixXd M = to_eigen(*t);
                    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> rows = M;
                    eigen_flat.push_back(
                        torch::from_blob(const_cast<double *>(rows.data()), {M.size()},
                                         torch::TensorOptions().dtype(torch::kDouble))
                            .clone());
                }
            }
        check(max_diff(flatten(with_torch), torch::cat(eigen_flat)) < 1e-10,
              where + ": " + name(algorithm) + ": libtorch and Eigen disagree");
    }
}

/// One call over a batch of sets, of systems, or of both equals one call per member.
void a_batch_equals_its_members(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 4, S = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5;
    const torch::Tensor cs = torch::randn({S, n}, opts), Gs = torch::randn({S, n, m}, opts);

    for (const Algorithm algorithm : kAlgorithms) {
        const std::string what = where + ": " + name(algorithm);

        // A batch of sets under one system.
        const Reach sets = LinearSys(from_torch(A)).reach(make_set(cs, Gs), 0.1, 1.0, 8, algorithm);
        for (int64_t b = 0; b < S; ++b) {
            const Reach one =
                LinearSys(from_torch(A)).reach(make_set(cs[b], Gs[b]), 0.1, 1.0, 8, algorithm);
            for (std::size_t k = 0; k < one.time_int.size(); ++k) {
                check(max_diff(to_torch(sets.time_int[k].c)[b], to_torch(one.time_int[k].c)) <
                              1e-12 &&
                          max_diff(to_torch(sets.time_int[k].G)[b], to_torch(one.time_int[k].G)) <
                              1e-12,
                      what + ": a batched set differs from its own run");
            }
        }

        // A batch of systems from one set, and a batch of each together.
        const Reach systems =
            LinearSys(from_torch(As)).reach(make_set(cs[0], Gs[0]), 0.1, 1.0, 8, algorithm);
        const Reach both =
            LinearSys(from_torch(As)).reach(make_set(cs, Gs), 0.1, 1.0, 8, algorithm);
        for (int64_t b = 0; b < S; ++b) {
            const Reach one_sys =
                LinearSys(from_torch(As[b])).reach(make_set(cs[0], Gs[0]), 0.1, 1.0, 8, algorithm);
            const Reach one_both =
                LinearSys(from_torch(As[b])).reach(make_set(cs[b], Gs[b]), 0.1, 1.0, 8, algorithm);
            for (std::size_t k = 0; k < one_sys.time_int.size(); ++k) {
                check(max_diff(to_torch(systems.time_int[k].G)[b],
                               to_torch(one_sys.time_int[k].G)) < 1e-12,
                      what + ": a batched system differs from its own run");
                check(max_diff(to_torch(both.time_int[k].G)[b], to_torch(one_both.time_int[k].G)) <
                          1e-12,
                      what + ": a batch of both differs from its own run");
            }
        }
    }
}

/// A scalar of the reachable tube: the support of every step's enclosure along `d`.
torch::Tensor loss(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G,
                   Algorithm algorithm, bool custom_backward, const torch::Tensor &d) {
    const Reach r = LinearSys(from_torch(A, custom_backward))
                        .reach(make_set(c, G, custom_backward), 0.1, 1.0, 6, algorithm);
    torch::Tensor total = torch::zeros({}, A.options());
    for (const Zonotope &Z : r.time_int)
        total =
            total + to_torch(Z.support_func(from_torch(d.unsqueeze(-1), custom_backward))).sum();
    return total;
}

/// Gradients through the whole reach computation against central differences, and the
/// hand-written backward pass of the matrix exponential against autograd's own.
void the_gradients_are_right(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 3;
    const torch::Tensor d = torch::randn({n}, opts);
    for (const Algorithm algorithm : kAlgorithms) {
        const std::string what = where + ": " + name(algorithm);
        const torch::Tensor A0 = torch::randn({n, n}, opts) * 0.5;
        const torch::Tensor c0 = torch::randn({n}, opts), G0 = torch::randn({n, m}, opts);

        std::vector<torch::Tensor> grads[2];
        for (const bool custom : {false, true}) {
            torch::Tensor A = A0.clone().requires_grad_(true), c = c0.clone().requires_grad_(true),
                          G = G0.clone().requires_grad_(true);
            const torch::Tensor L = loss(A, c, G, algorithm, custom, d);
            grads[custom] = torch::autograd::grad({L}, {A, c, G});
        }
        for (int i = 0; i < 3; ++i)
            check(max_diff(grads[0][i], grads[1][i]) < 1e-9,
                  what + ": the custom backward differs from autograd's (input " +
                      std::to_string(i) + ")");

        // Central differences over every entry of A, c and G.
        const double h = 1e-6;
        auto numeric = [&](torch::Tensor A, torch::Tensor c, torch::Tensor G, int which,
                           int64_t at) {
            torch::Tensor *t = which == 0 ? &A : which == 1 ? &c : &G;
            torch::Tensor up = t->clone(), down = t->clone();
            up.view({-1})[at] += h;
            down.view({-1})[at] -= h;
            auto run = [&](const torch::Tensor &v) {
                torch::NoGradGuard no_grad;
                return loss(which == 0 ? v : A, which == 1 ? v : c, which == 2 ? v : G, algorithm,
                            false, d)
                    .item<double>();
            };
            return (run(up) - run(down)) / (2 * h);
        };
        const torch::Tensor tensors[3] = {A0, c0, G0};
        double worst = 0.0;
        for (int which = 0; which < 3; ++which)
            for (int64_t at = 0; at < tensors[which].numel(); ++at) {
                const double want = numeric(A0, c0, G0, which, at);
                const double got = grads[0][which].to(torch::kCPU).view({-1})[at].item<double>();
                worst = std::max(worst, std::abs(want - got) / (1.0 + std::abs(want)));
            }
        check(worst < 1e-5,
              what + ": a gradient is off from central differences by " + std::to_string(worst));
    }
}

/// The same run on the device and on the host.
void the_device_agrees_with_the_host(const torch::TensorOptions &device_opts) {
    const auto host = torch::TensorOptions().dtype(torch::kDouble);
    const int64_t n = 5, m = 4;
    const torch::Tensor A = torch::randn({2, n, n}, host) * 0.5;
    const torch::Tensor c = torch::randn({2, n}, host), G = torch::randn({2, n, m}, host);
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach on_host =
            LinearSys(from_torch(A)).reach(make_set(c, G), 0.1, 1.0, 8, algorithm);
        const Reach on_device =
            LinearSys(from_torch(A.to(device_opts.device())))
                .reach(make_set(c.to(device_opts.device()), G.to(device_opts.device())), 0.1, 1.0,
                       8, algorithm);
        check(max_diff(flatten(on_host), flatten(on_device)) < 1e-9,
              "gpu: " + name(algorithm) + " differs from the host");
    }
}

/// The same code, written with `Tensor` constructors only, gives the same answer whichever
/// backend was set â€” the point of setting it in one place.
void the_backend_is_set_in_one_place() {
    auto run = [] {
        const Tensor A({{-0.2, 1.0}, {-1.0, -0.2}});
        const Zonotope X0{Tensor({1.0, 0.5}), Tensor({{0.1, 0.0}, {0.0, 0.2}})};
        const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 8, Algorithm::Standard);
        return R.time_int.back().support_func(Tensor({1.0, 1.0})).data();
    };
    set_backend("eigen");
    const std::vector<double> eigen = run();
    set_backend("torch");
    const std::vector<double> libtorch = run();
    check(eigen.size() == 1 && libtorch.size() == 1 && std::abs(eigen[0] - libtorch[0]) < 1e-10,
          "set_backend: eigen and torch runs of the same code differ");

    // Tensors of different backends do not mix.
    set_backend("eigen");
    const Tensor on_eigen({1.0, 2.0});
    set_backend("torch");
    const Tensor on_torch({1.0, 2.0});
    bool threw = false;
    try {
        (void)(on_eigen + on_torch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "set_backend: tensors of two backends were combined");

    bool rejected = false;
    try {
        set_backend("jax");
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "set_backend: an unknown backend was accepted");
}

} // namespace

int main() {
    std::vector<std::string> devices{"cpu"};
    if (torch::cuda::is_available()) devices.emplace_back("gpu");

    for (const std::string &device : devices) {
        const auto opts = torch::TensorOptions()
                              .dtype(torch::kDouble)
                              .device(device == "gpu" ? torch::Device(torch::kCUDA, 0)
                                                      : torch::Device(torch::kCPU));
        torch::manual_seed(7);
        the_backends_agree(opts, device);
        a_batch_equals_its_members(opts, device);
        the_gradients_are_right(opts, device);
        if (device == "gpu") the_device_agrees_with_the_host(opts);
    }
    the_backend_is_set_in_one_place();
    if (devices.size() == 1)
        std::cout << "no CUDA device: the gpu half of these tests did not run\n";
    if (failures == 0) std::cout << "all libtorch linearSys tests passed\n";
    return failures == 0 ? 0 : 1;
}
