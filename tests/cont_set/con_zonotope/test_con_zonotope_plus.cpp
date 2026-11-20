// test_con_zonotope_plus - ConZonotope::plus: the Minkowski sum
//
// CORA: contSet/@conZonotope/plus.m
// PROPERTY: contSet/@conZonotope/plus.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: centers add, generators join, constraints on the block diagonal; a point moves
/// the center; a zonotope and a box are sets without constraints.
void plus_known(const std::string &b) {
    const ConZonotope a(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        vec({0.5}));
    const ConZonotope s(column({10.0, 20.0}), Tensor({{2.0, 3.0}}), Tensor({{4.0}}), vec({1.0}));
    const ConZonotope r = a.plus(s);
    check(close(r.c().expand_dims(-1), {11, 22}, 0), b + ": the centers add");
    check(close(r.generators().matrix_transpose(), {1, 0, 2, 0, 1, 3}, 0),
          b + ": the generators join");
    check(close(r.A(), {1, 1, 0, 0, 0, 4}, 0) && close(r.b(), {0.5, 1}, 0),
          b + ": the constraints on the block diagonal");
    check(close(a.plus(vec({1.0, 1.0})).c().expand_dims(-1), {2, 3}, 0),
          b + ": a point moves the center");
    const Zonotope Z(column({1.0, 1.0}), Tensor({{1.0, 0.0}}));
    check(close(a.plus(Z).generators().matrix_transpose(), {1, 0, 1, 0, 1, 0}, 0) &&
              a.plus(Z).A().shape()[1] == 3,
          b + ": a zonotope joins as a set without constraints");
    // a box is a set without constraints whose generators are its radius
    const Interval I(vec({0.0, 0.0}), vec({2.0, 4.0}));
    check(close(a.plus(I).c().expand_dims(-1), {2, 4}, 0), b + ": a box adds its center");
    check(close((a + vec({1.0, 1.0})).c().expand_dims(-1), {2, 3}, 0) &&
              close((vec({1.0, 1.0}) + a).c().expand_dims(-1), {2, 3}, 0) &&
              close((a - vec({1.0, 1.0})).c().expand_dims(-1), {0, 1}, 0),
          b + ": the operators");
    const ConZonotope negated = Set(a)->uminus().as<ConZonotope>();
    check(close(negated.c().expand_dims(-1), {-1, -2}, 0) &&
              close(negated.generators().matrix_transpose(),
                    a.generators().matrix_transpose() * -1.0, 0) &&
              close(negated.A(), a.A(), 0),
          b + ": uminus negates the center and the generators");
}

/// Stated property: sums of sampled points of the two sets are in the sum.
void plus_property(const std::string &b) {
    cora::Rng rng(41);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope p = test::random_con_zonotope(rng, 3, 4, 1);
        const ConZonotope q = test::random_con_zonotope(rng, 3, 5, 2);
        const ConZonotope r = p.plus(q);
        const Tensor sums = test::sample_points(p, rng, 20) + test::sample_points(q, rng, 20);
        check(r.contains(sums.matrix_transpose()), b + ": every sum of points is in the sum");
        check(close(r.interval().sup(), p.interval().sup() + q.interval().sup(), 1e-8),
              b + ": the box of the sum is the sum of the boxes");
    }
}

/// A different dimension and a point that is no column are refused; a batch adds member by member.
void checks_and_batches() {
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope one(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.plus(one); }, Error::dimension_mismatch), "plus: dimensions");
    check(test::raises([&] { a.plus(vec({1.0})); }, Error::dimension_mismatch),
          "plus: a point of another dimension");
    const ConZonotope batch(Tensor::zeros({2, 2, 1}), Tensor::zeros({2, 1, 2}));
    check(test::raises([&] { a.plus(batch); }, Error::dimension_mismatch), "plus: batch shapes");
    check(batch.plus(batch).c().expand_dims(-1).shape() == Shape{2, 2, 1},
          "plus: a batch adds member by member");
    ChecksScope off(false);
    check(a.plus(a).generators().matrix_transpose().shape() == Shape{2, 4},
          "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        plus_known(b);
        plus_property(b);
    });
    checks_and_batches();
    return test::finish("con_zonotope plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
