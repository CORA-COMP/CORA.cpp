// test_zonotope_compact - zonotope compact: drops zero generators, merges aligned ones
//
// CORA: contSet/@zonotope/compact_.m
// PROPERTY: contSet/@zonotope/compact_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// "zeros" drops the generators of norm <= tol: exactly zero ones, tiny ones by tol.
void zeros_drops_small_generators(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 1e-20, 0.5}, {0.0, 0.0, 0.0, 0.5}}));
    check(close(Z.compact().generators(), std::vector<double>{1, 0.5, 0, 0.5}, 0),
          b + ": zeros drops the zero and the tiny generator (norm <= eps)");
    check(Z.compact("zeros", 0.0).generators().shape().back() == 3, b + ": tol 0 keeps the tiny one");
    check(Z.compact("zeros", 0.6).generators().shape().back() == 2, b + ": tol 0.6 keeps two");
    check(Z.compact("zeros", 0.8).generators().shape().back() == 1, b + ": tol 0.8 keeps one");
    check(close(Z.compact().center(), Z.center(), 0), b + ": the center stays");
    const Zonotope none(column({1.0}), Tensor({{0.0, 0.0}}));
    check(none.compact().generators().shape().back() == 0, b + ": all generators zero leaves none");
}

/// "all" merges parallel and antiparallel generators into the shortest (CORA's aux_deleteAligned).
void all_merges_aligned_generators(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{-0.5, 1.0, 2.0, 0.0}, {0.0, 0.0, 0.0, 3.0}}));
    const Zonotope C = Z.compact("all");
    check(close(C.generators(), std::vector<double>{-3.5, 0, 0, 3}, 1e-12),
          b + ": (-0.5,0), (1,0), (2,0) merge into (-3.5,0)");
    const Zonotope one(column({0.0}), Tensor({{1.0, -2.0, 0.0}}));
    check(close(one.compact("all").generators(), std::vector<double>{3}, 0),
          b + ": in one dimension the generators add up");
    const Zonotope diff(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(close(diff.compact("all").generators(), diff.generators(), 0),
          b + ": generators that are not aligned stay");
    check(close(Z.compact("aligned").generators(), Z.generators(), 0),
          b + ": aligned leaves the zonotope as it is, as CORA");
}

/// A batch drops the generators that every member drops.
void a_batch_keeps_what_some_member_keeps(const std::string &b) {
    const Zonotope Z1(column({0.0, 0.0}), Tensor({{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}));
    const Zonotope Z2(column({0.0, 0.0}), Tensor({{0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}}));
    const Zonotope C = Zonotope::stack({Z1, Z2}).compact();
    check(C.generators().shape() == std::vector<int64_t>({2, 2, 2}),
          b + ": the middle generator is zero in both members, the others stay");
    check(close(C.generators().slice(0, 1, 2).squeeze(0), std::vector<double>{0, 0, 0, 2}, 0),
          b + ": member 2 keeps its generator");
}

/// Stated property: compact keeps the set (the support function) for "zeros" and "all".
void compact_keeps_the_set(const std::string &b) {
    cora::Rng rng(14);
    const int n = 3;
    Zonotope Z = Zonotope::generate_random(n, 5, rng);
    const Tensor g = Z.generators();
    // repeated and opposite generators and a zero one
    const Zonotope R(Z.center(), Tensor::concat({g, g.slice(-1, 0, 1) * -2.0, g * 0.0,
                                                  g.slice(-1, 1, 2) * 3.0}, -1));
    for (const std::string method : {"zeros", "all"}) {
        const Zonotope C = R.compact(method, 1e-10);
        check(C.generators().shape().back() < R.generators().shape().back(),
              b + ": " + method + " removes generators");
        for (int k = 0; k < 30; ++k) {
            const std::vector<double> d = test::random_direction(rng, n);
            check(test::close(test::support(C, d), test::support(R, d), 1e-9),
                  b + ": " + method + " keeps the support function");
        }
    }
}

/// Wrong arguments are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.0}}));
    check(test::raises([&] { Z.compact("sorted"); }, Error::wrong_value), "compact: unknown method");
    check(test::raises([&] { Z.compact("zeros", -1.0); }, Error::wrong_value), "compact: tol < 0");
    check(test::raises([&] { Zonotope::stack({Z, Z}).compact("all"); }, Error::not_supported),
          "compact: all for a batch");
    const Tensor on = Z.compact().generators();
    ChecksScope off(false);
    check(test::close(Z.compact().generators(), on, 0), "compact: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        zeros_drops_small_generators(b);
        all_merges_aligned_generators(b);
        a_batch_keeps_what_some_member_keeps(b);
        compact_keeps_the_set(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope compact");
}
