// test_tensor_copy_on_write - an operation on a tensor nobody else holds reuses its memory
//
// A tensor passed as an rvalue (std::move(t) + 1, std::move(t).abs()) that owns its memory alone is
// written in place; a shared one is copied first, so no other holder sees the change. The standard
// linear reach allocates a fixed number of tensors per step on eigen, counted here by the global
// operator new (Eigen's own aligned buffers are not counted; the fused kernels keep none per step).

#include "cora/cora.h"
#include "cora/global/backend/host.h"
#include "testing.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

std::atomic<long long> aux_allocations{0};

} // namespace

// Every tensor's memory and every shared state comes from here: count them.
void *operator new(std::size_t n) {
    ++aux_allocations;
    if (void *p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

using namespace cora;
using test::check;

namespace {

/// The address of a host tensor's entries.
const void *aux_address(const Tensor &t) { return host_view(t).data; }

void an_unshared_tensor_is_written_in_place() {
    const BackendScope scope("eigen");
    Tensor t = Tensor::ones({10, 10});
    const void *before = aux_address(t);
    t = std::move(t) + 1.0;
    check(aux_address(t) == before, "t + 1 of an unshared t keeps its memory");
    t = std::move(t).abs();
    check(aux_address(t) == before, "abs of an unshared t keeps its memory");
    t = std::move(t) * 1.0;
    check(aux_address(t) == before, "a scaling of an unshared t keeps its memory");
    check(test::close(t, Tensor::ones({10, 10}) * 2.0, 0), "and computes 2");
}

void a_shared_tensor_is_copied_first() {
    const BackendScope scope("eigen");
    Tensor t = Tensor::ones({10, 10});
    const Tensor other = t;
    t = std::move(t) + 1.0;
    check(aux_address(t) != aux_address(other), "a shared t gets new memory");
    check(test::close(other, Tensor::ones({10, 10}), 0), "the other holder still sees ones");
}

void the_standard_reach_allocates_a_fixed_number_per_step() {
    const BackendScope scope("eigen");
    const LinearSys sys(Tensor({{-1, -4, 0, 0, 0}, {4, -1, 0, 0, 0}, {0, 0, -3, 1, 0},
                                {0, 0, -1, -3, 0}, {0, 0, 0, 0, -2}}));
    const Zonotope X0(Tensor({1, 1, 1, 1, 1}), Tensor::eye(5) * 0.1);
    long long counts[2];
    for (int k = 0; k < 2; ++k) {
        const long long before = aux_allocations.load();
        const ReachSet R = sys.reach({.initial_set = X0, .t_final = 2.0 * (k + 1)},
                                     alg::Standard{.time_step = 0.02, .taylor_terms = 4});
        counts[k] = aux_allocations.load() - before;
    }
    // 100 more steps. A set is 3 allocations (two tensors; a center of 5 entries is inline): per
    // step the two kept sets, and the parts H and C of the enclosure H + C
    const double per_step = double(counts[1] - counts[0]) / 100.0;
    std::cout << "allocations per step: " << per_step << "\n";
    check(per_step <= 12.5, "the standard reach allocates at most 12 times per step");
}

} // namespace

int main() {
    an_unshared_tensor_is_written_in_place();
    a_shared_tensor_is_copied_first();
    the_standard_reach_allocates_a_fixed_number_per_step();
    return test::finish("tensor copy on write");
}
