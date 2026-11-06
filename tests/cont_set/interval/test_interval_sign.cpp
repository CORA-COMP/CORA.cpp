// test_interval_sign - interval sign: the hull of the sign of the bounds
//
// CORA: contSet/@interval/sign.m
// PROPERTY: contSet/@interval/sign.m
// CORA authors: Sebastian Mair, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void sign_of_the_bounds(const std::string &b) {
    const Interval I(column({-3.0, 1.0, 0.0, -2.0, 0.0}), column({2.0, 3.0, 2.0, 0.0, 0.0}));
    const Interval R = I.sign();
    check(R.inf().data() == std::vector<double>({-1, 1, 0, -1, 0}) &&
              R.sup().data() == std::vector<double>({1, 1, 1, 0, 0}),
          b + ": sign of the bounds");
    const Interval S = Interval::stack({I, I}).sign();
    check(S.inf().shape() == std::vector<int64_t>({2, 5, 1}), b + ": a batch has a batch shape");
    check(S.sup().slice(0, 1, 2).reshape({5, 1}).data() == R.sup().data(), b + ": a batch member");
}

/// The sign of every sampled point lies in the result, which is tight at the bounds.
void sign_encloses_the_image(const std::string &b) {
    cora::Rng rng(4);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(rng, 3), R = I.sign();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).sign());
        check(inside, b + ": the sign of a sample is in the result");
        check(R.inf().data() == I.inf().sign().data() && R.sup().data() == I.sup().sign().data(),
              b + ": the bounds are the signs of the bounds");
    }
}

/// sign has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.5}), column({1.0, 2.0}));
    const std::vector<double> on = I.sign().inf().data();
    ChecksScope off(false);
    check(I.sign().inf().data() == on, "sign: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sign_of_the_bounds(b);
        sign_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval sign");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
