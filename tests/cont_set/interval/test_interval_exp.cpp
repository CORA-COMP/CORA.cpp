// test_interval_exp - interval exp: the exponential of a box
//
// CORA: contSet/@interval/exp.m
// PROPERTY: contSet/@interval/exp.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The bounds map as the function does; a batch is mapped member by member.
void exp_maps_the_bounds(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({0.0, 2.0}));
    const Interval R = I.exp();
    check(close(R.inf(), std::vector<double>{std::exp(-1.0), 1.0}) &&
              close(R.sup(), std::vector<double>{1.0, std::exp(2.0)}),
          b + ": exp of the bounds");
    const Interval S = Interval::stack({I, I}).exp();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()), b + ": a batch member");
}

/// The image of every sampled point of the box lies in the result, which is tight at the bounds.
void exp_encloses_the_image(const std::string &b) {
    cora::Rng rng(31);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> lo(3), hi(3);
        rng.uniform(lo.data(), 3, -4, 4);
        rng.uniform(hi.data(), 3, -4, 4);
        for (int i = 0; i < 3; ++i)
            if (lo[i] > hi[i]) std::swap(lo[i], hi[i]);
        const Interval I(column(lo), column(hi)), R = I.exp();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).exp());
        check(inside, b + ": exp of a sample is in the result");
        check(close(R.inf(), I.inf().exp(), 0) && close(R.sup(), I.sup().exp(), 0),
              b + ": the bounds are the images of the bounds");
    }
}

/// exp has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.exp().sup();
    ChecksScope off(false);
    check(close(I.exp().sup(), on, 0), "exp: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        exp_maps_the_bounds(b);
        exp_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval exp");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
