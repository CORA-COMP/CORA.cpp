// test_interval_zonotope - interval zonotope: the zonotope of a box
//
// CORA: contSet/@interval/zonotope.m
// PROPERTY: contSet/@interval/zonotope.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void zonotope_has_the_radius_as_generators(const std::string &b) {
    const Interval I(column({0.0, 1.0, -1.0}), column({2.0, 1.0, 3.0}));
    const Zonotope Z = I.zonotope();
    check(close(Z.center(), std::vector<double>{1, 1, 1}), b + ": the center");
    check(Z.generators().shape() == std::vector<int64_t>({3, 2}) &&
              close(Z.generators(), std::vector<double>{1, 0, 0, 0, 0, 2}),
          b + ": diag(r) without the zero-radius column");
    const Interval P(column({1.0, 2.0}), column({1.0, 2.0}));
    check(P.zonotope().generators().shape() == std::vector<int64_t>({2, 0}),
          b + ": a point has no generators");
    const Zonotope S = Interval::stack({I, I}).zonotope();
    check(S.generators().shape() == std::vector<int64_t>({2, 3, 2}),
          b + ": a batch drops the columns of zero radius in every member");
    const Interval J(column({0.0, 1.0, -1.0}), column({2.0, 1.0, -1.0}));
    check(Interval::stack({I, J}).zonotope().generators().shape() == std::vector<int64_t>({2, 3, 2}),
          b + ": a batch keeps a column that has a radius in some member");
}

/// The zonotope has the box as its interval hull, and both contain the same sampled points.
void zonotope_is_the_box(const std::string &b) {
    cora::Rng rng(121);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(4, rng);
        const Zonotope Z = I.zonotope();
        const Interval H = Z.interval();
        check(close(H.inf(), I.inf(), 1e-12) && close(H.sup(), I.sup(), 1e-12),
              b + ": the interval hull of the zonotope is the box");
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && Z.contains(x.slice(-1, k, k + 1));
        check(inside, b + ": a sample of the box is in the zonotope");
    }
}

/// A matrix and the empty interval are refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    check(test::raises([&] { Interval(lo, lo + 1.0).zonotope(); }, Error::wrong_value),
          "zonotope: a matrix");
    check(test::raises([] { Interval::empty(2).zonotope(); }, Error::not_supported),
          "zonotope: the empty interval");
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    const Tensor on = I.zonotope().generators();
    ChecksScope off(false);
    check(close(I.zonotope().generators(), on, 0), "zonotope: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        zonotope_has_the_radius_as_generators(b);
        zonotope_is_the_box(b);
    });
    checks_and_their_switch();
    return test::finish("interval zonotope");
}
