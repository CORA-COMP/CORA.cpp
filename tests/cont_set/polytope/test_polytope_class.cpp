// test_polytope_class - the constructors of Polytope: forms, processing of the data and refusals
//
// CORA: contSet/@polytope/polytope.m
// CORA: contSet/@interval/polytope.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The halfspace forms, the vertex form and the processing the constructor does.
void constructor_forms(const std::string &b) {
    const Tensor A({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}});
    const Polytope P(A, column({1.0, 2.0, 3.0, 4.0}));
    check(P.dim() == 2 && P.is_h_rep() && !P.is_v_rep() && P.class_name() == "Polytope",
          b + ": halfspaces: dimension and form");
    check(P.Ae().shape() == Shape{0, 2} && P.be().shape() == Shape{0, 1},
          b + ": without equalities Ae has no rows");
    const Polytope with_equality(A, column({1.0, 2.0, 3.0, 4.0}), Tensor({{1.0, 1.0}}),
                                 column({0.5}));
    check(close(with_equality.Ae(), {1, 1}, 0) && close(with_equality.be(), {0.5}, 0),
          b + ": the equalities");
    // vertices
    const Polytope V(Tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    check(V.dim() == 2 && V.is_v_rep() && !V.is_h_rep() && V.V().shape() == Shape{2, 3},
          b + ": vertices: form and shape");
    // in 1D only the least and the greatest vertex are kept
    const Polytope line(Tensor({{3.0, 1.0, 2.0, 5.0}}));
    check(close(line.V(), {1, 5}, 0), b + ": 1D keeps the extremes");
    check(close(Polytope(Tensor({{2.0, 2.0}})).V(), {2}, 0), b + ": 1D: equal extremes are one");
    const Polytope unbounded_line(Tensor({{-std::numeric_limits<double>::infinity(), 1.0}}));
    check(unbounded_line.V().shape() == Shape{1, 2}, b + ": 1D: vertices may be infinite");
    // a polytope of no vertices is the empty set
    check(Polytope(Tensor::zeros({3, 0})).V().shape() == Shape{3, 0}, b + ": no vertices");
    // rows with an infinite offset hold everywhere and go; an offset -Inf is the empty set
    const double inf = std::numeric_limits<double>::infinity();
    const Polytope dropped(A, column({1.0, inf, 3.0, 4.0}));
    check(dropped.A().shape() == Shape{3, 2} && close(dropped.b(), {1, 3, 4}, 0),
          b + ": +Inf offsets are dropped");
    const Polytope infeasible(A, column({1.0, -inf, 3.0, 4.0}));
    check(infeasible.is_h_rep() && infeasible.is_v_rep() && infeasible.V().shape() == Shape{2, 0},
          b + ": a -Inf offset: the empty set without vertices");
    check(Polytope(A, column({1.0, 2.0, 3.0, 4.0}), Tensor({{1.0, 1.0}}), column({inf}))
                  .V()
                  .shape() == Shape{2, 0},
          b + ": an infinite equality offset: the empty set");
    // a Set holds it as its class
    const Set via_set = P;
    check(via_set.is<Polytope>() && via_set.class_id() == ClassId::polytope,
          b + ": a Set holds it as its class");
}

/// A box as a polytope.
void from_interval(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({2.0, 3.0}));
    const Polytope P(I);
    check(close(P.A(), {1, 0, 0, 1, -1, 0, 0, -1}, 0) && close(P.b(), {2, 3, 0, 1}, 0),
          b + ": the rows x <= sup, -x <= -inf");
    const double inf = std::numeric_limits<double>::infinity();
    const Polytope half(Interval(column({0.0, -inf}), column({inf, 3.0})));
    check(close(half.A(), {0, 1, -1, 0}, 0) && close(half.b(), {3, 0}, 0),
          b + ": unbounded sides give no row");
    check(Polytope(Interval::unbounded(2)).A().shape() == Shape{0, 2}, b + ": a box of R^2 is R^2");
    check(Polytope(Interval::empty(2)).V().shape() == Shape{2, 0}, b + ": the empty box is empty");
}

/// What the accessors refuse and what the constructors refuse.
void refusals() {
    const Tensor A({{1.0, 0.0}, {0.0, 1.0}});
    const Polytope P(A, column({1.0, 2.0}));
    check(test::raises([&] { P.V(); }, Error::special_error), "V: only the halfspaces are held");
    const Polytope V(Tensor({{0.0, 1.0}, {0.0, 1.0}}));
    check(test::raises([&] { V.A(); }, Error::special_error), "A: only the vertices are held");
    check(test::raises([&] { V.b(); }, Error::special_error), "b: only the vertices are held");
    check(test::raises([&] { V.Ae(); }, Error::special_error), "Ae: only the vertices are held");
    check(test::raises([&] { V.be(); }, Error::special_error), "be: only the vertices are held");
    check(test::raises([&] { Polytope(A, column({1.0})); }, Error::wrong_input_in_constructor),
          "Polytope: the rows of A and b");
    check(
        test::raises([&] { Polytope(A, Tensor({{1.0, 2.0}})); }, Error::wrong_input_in_constructor),
        "Polytope: b is a column");
    check(test::raises(
              [&] { Polytope(A, column({1.0, 2.0}), Tensor({{1.0, 1.0, 1.0}}), column({0.0})); },
              Error::wrong_input_in_constructor),
          "Polytope: the columns of A and Ae");
    check(test::raises(
              [&] { Polytope(A, column({1.0, 2.0}), Tensor({{1.0, 1.0}}), column({0.0, 1.0})); },
              Error::wrong_input_in_constructor),
          "Polytope: the rows of Ae and be");
    check(test::raises([&] { Polytope(Tensor({{0.0, std::nan("")}})); },
                       Error::wrong_input_in_constructor),
          "Polytope: vertices must not be NaN");
    check(
        test::raises(
            [&] { Polytope(Tensor({{0.0, std::numeric_limits<double>::infinity()}, {0.0, 1.0}})); },
            Error::wrong_input_in_constructor),
        "Polytope: vertices must be finite for n > 1");
    check(test::raises([&] { Polytope(Tensor::zeros({3})); }, Error::wrong_input_in_constructor),
          "Polytope: the vertices are a matrix");
    // with the checks off the same valid call builds the same set
    const Polytope on(A, column({1.0, 2.0}));
    ChecksScope off(false);
    const Polytope same(A, column({1.0, 2.0}));
    check(close(same.A(), on.A(), 0) && close(same.b(), on.b(), 0),
          "Polytope: the same with the checks off");
}

/// A batch of polytopes shares one batch shape.
void batches(const std::string &b) {
    const Tensor A = Tensor::ones({3, 2, 2}), bb = Tensor::ones({3, 2, 1});
    const Polytope P(A, bb);
    check(P.dim() == 2 && P.A().shape() == Shape{3, 2, 2} && P.Ae().shape() == Shape{3, 0, 2},
          b + ": a batch of three polytopes");
    check(test::raises([&] { Polytope(A, Tensor::ones({2, 2, 1})); },
                       Error::wrong_input_in_constructor),
          b + ": one batch shape");
    const double inf = std::numeric_limits<double>::infinity();
    check(test::raises([&] { Polytope(A, Tensor::ones({3, 2, 1}) * inf); }, Error::not_supported),
          b + ": an infinite offset in a batch is refused");
    const Polytope V(Tensor::ones({4, 2, 3}));
    check(V.V().shape() == Shape{4, 2, 3}, b + ": a batch of vertex polytopes");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        constructor_forms(b);
        from_interval(b);
        batches(b);
    });
    refusals();
    return test::finish("polytope class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
