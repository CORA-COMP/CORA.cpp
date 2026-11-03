// test_zonotope_reduce_under_approx - zonotope reduce_under_approx: a zonotope inside with fewer
// generators
//
// CORA: contSet/@zonotope/reduceUnderApprox.m
// PROPERTY: contSet/@zonotope/reduceUnderApprox.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <tuple>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// CORA's unit test: eight generators in two dimensions reduced to order 1.
Zonotope example() {
    return {column({1.0, 0.0}), Tensor({{1.0, 3.0, -2.0, 3.0, -1.0, 0.0, 2.0, 3.0},
                                       {2.0, -1.0, 1.0, 0.0, 3.0, -2.0, 1.0, 2.0}})};
}

/// kochdumper by hand: the longest generator (3, 2) stays, the others (signed by their first
/// entry; the one with first entry 0 drops out) add to (12, -2).
void kochdumper_by_hand(const std::string &b) {
    for (const char *name : {"kochdumper", "sum"}) {
        const Zonotope R = example().reduce_under_approx(name, 1);
        check(close(R.center(), std::vector<double>{1, 0}, 0), b + ": the center stays");
        check(close(R.generators(), std::vector<double>{3, 12, 2, -2}, 1e-14),
              b + ": " + name + ": generators (3, 2) and (12, -2)");
    }
}

/// raghuraman by hand, as CORA's code: the two longest generators are (3, 2) and (3, -1); the six
/// others are attached with the sign of their product to the big one they are most aligned
/// with, the sign pattern is applied to the generators in their original order.
void raghuraman_by_hand(const std::string &b) {
    const Zonotope R = example().reduce_under_approx("raghuraman", 1);
    check(close(R.generators(), std::vector<double>{2, 5, 4, 0}, 1e-14),
          b + ": generators (2, 4) and (5, 0)");
    // a small generator orthogonal to the big one: CORA divides 0/0 and gets NaN, this drops it
    const Zonotope Z(column({0.0, 0.0}), Tensor({{3.0, 0.0}, {0.0, 1.0}}));
    check(close(Z.reduce_under_approx("raghuraman", 0.5).generators(), std::vector<double>{3, 0}, 0),
          b + ": an orthogonal small generator is dropped");
}

/// Few enough generators: the zonotope without its zero generators.
void nothing_to_reduce(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0, 2.0}, {0.0, 0.0, 1.0}}));
    for (const char *name : {"yang", "raghuraman", "kochdumper"}) {
        const Zonotope R = Z.reduce_under_approx(name, 1);
        check(close(R.generators(), std::vector<double>{1, 2, 0, 1}, 0),
              b + ": " + name + ": two generators in two dimensions stay");
    }
}

/// Stated property: the result has at most order * n generators and lies in Z (CORA's test:
/// contains with tol 1e-10), for random zonotopes and orders.
void result_is_inside(const std::string &b) {
    cora::Rng rng(121);
    for (const char *name : {"yang", "raghuraman", "kochdumper"})
        for (const auto [n, m, order] : {std::tuple{2, 8, 1.0}, {3, 9, 2.0}, {2, 12, 2.5}}) {
            const Zonotope Z = Zonotope::generate_random(n, m, rng);
            const Zonotope R = Z.reduce_under_approx(name, order);
            check(double(R.generators().shape().back()) <= order * n,
                  b + ": " + name + ": at most order * n generators");
            check(close(R.center(), Z.center(), 0), b + ": " + name + ": the center");
            check(Z.contains(R, "exact", 1e-10), b + ": " + name + ": the result is inside Z");
        }
}

/// Unsupported methods, unknown ones, a bad order and a batch are refused; with the checks off
/// the result is the same.
void checks_and_their_switch() {
    const Zonotope Z = example();
    for (const char *name : {"sadraddini", "scale", "boxLP", "boxCone", "boxWetzlinger", "nlp",
                             "cluster", "linProg", "wetzlinger"})
        check(test::raises([&] { Z.reduce_under_approx(name, 1); }, Error::not_supported),
              std::string("reduce_under_approx: ") + name);
    check(test::raises([&] { Z.reduce_under_approx("boxNLP", 1); }, Error::wrong_value), "an unknown method");
    check(test::raises([&] { Z.reduce_under_approx("yang", 0); }, Error::wrong_value), "order 0");
    check(test::raises([&] { Z.reduce_under_approx("yang", -1); }, Error::wrong_value), "order -1");
    check(test::raises([&] { Z.reduce_under_approx("kochdumper", 0.4); }, Error::wrong_value),
          "an order that leaves no generator for the sum");
    check(test::raises([&] { Z.reduce_under_approx("raghuraman", 0.4); }, Error::wrong_value),
          "an order that leaves no generator");
    check(test::raises([&] { Zonotope::stack({Z, Z}).reduce_under_approx("yang", 1); },
                       Error::not_supported),
          "a batch");
    const Tensor on = Z.reduce_under_approx("yang", 1).generators();
    ChecksScope off(false);
    check(close(Z.reduce_under_approx("yang", 1).generators(), on, 0), "checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        kochdumper_by_hand(b);
        raghuraman_by_hand(b);
        nothing_to_reduce(b);
        result_is_inside(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope reduce_under_approx");
}
