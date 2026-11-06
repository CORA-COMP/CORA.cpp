// test_interval_atan - interval atan: the arctangent of a box
//
// CORA: contSet/@interval/atan.m
// PROPERTY: contSet/@interval/atan.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A box of n dimensions with bounds drawn from [low, high].
Interval aux_sample(cora::Rng &rng, int n, double low, double high) {
    std::vector<double> a(n), b(n);
    rng.uniform(a.data(), n, low, high);
    rng.uniform(b.data(), n, low, high);
    for (int i = 0; i < n; ++i)
        if (a[i] > b[i]) std::swap(a[i], b[i]);
    return {column(a), column(b)};
}

/// The bounds map as the function does; a batch is mapped member by member.
void atan_maps_the_bounds(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Interval R = I.atan();
    check(close(R.inf(), std::vector<double>{-0.7853981633974483, 0.0}) &&
              close(R.sup(), std::vector<double>{0.7853981633974483, 0.7853981633974483}),
          b + ": atan of the bounds");
    const Interval S = Interval::stack({I, I}).atan();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()), b + ": a batch member");
}

/// The image of every sampled point of the box lies in the result.
void atan_encloses_the_image(const std::string &b) {
    cora::Rng rng(33);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = aux_sample(rng, 3, -8, 8);
        const Interval R = I.atan();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).atan());
        check(inside, b + ": atan of a sample is in the result");
    }
}

/// atan has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.atan().sup();
    ChecksScope off(false);
    check(close(I.atan().sup(), on, 0), "atan: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        atan_maps_the_bounds(b);
        atan_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval atan");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
