// test_interval_uminus - interval uminus: the negated box
//
// CORA: contSet/@interval/uminus.m
// PROPERTY: contSet/@interval/uminus.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void uminus_swaps_and_negates(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({2.0, 0.5}));
    const Interval R = I.uminus();
    check(close(R.inf(), std::vector<double>{-2.0, -0.5}) &&
              close(R.sup(), std::vector<double>{1.0, -0.0}),
          b + ": [-sup, -inf]");
    check(close((-I).inf(), R.inf(), 0) && close((-I).sup(), R.sup(), 0), b + ": the operator");
    check(close(R.uminus().inf(), I.inf(), 0), b + ": twice is the identity");
    const Interval S = Interval::stack({I, I}).uminus();
    check(close(S.inf().slice(0, 1, 2).reshape({2, 1}), R.inf()), b + ": a batch member");
}

/// The negation of every sampled point lies in the result: its set is exactly {-x}.
void uminus_is_the_set_of_negations(const std::string &b) {
    cora::Rng rng(41);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(3, rng), R = -I;
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(-x.slice(-1, k, k + 1));
        check(inside, b + ": the negation of a sample is in the result");
        check(close(R.rad(), I.rad(), 0) && close(R.center(), -I.center(), 0),
              b + ": the same radius and the negated center");
    }
}

/// uminus has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    const Tensor on = I.uminus().sup();
    ChecksScope off(false);
    check(close(I.uminus().sup(), on, 0), "uminus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        uminus_swaps_and_negates(b);
        uminus_is_the_set_of_negations(b);
    });
    checks_and_their_switch();
    return test::finish("interval uminus");
}
