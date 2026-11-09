// test_linear_reset_evaluate - LinearReset: its constructor and the image of sets and points
//
// A reset map applied to a box, to a bundle and to points, by hand; the refused inputs.
//
// CORA: hybridDynamics/@linearReset/linearReset.m
// CORA: hybridDynamics/@linearReset/evaluate.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/hybrid_dynamics/linear_reset/linear_reset.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The reset (x, v) -> (x, -0.5 v) + (1, 0) of the box [0, 1] x [2, 4]: [1, 2] x [-2, -1].
void box_by_hand(const std::string &b) {
    const LinearReset reset(Tensor({{1.0, 0.0}, {0.0, -0.5}}), std::nullopt, column({1.0, 0.0}));
    const Zonotope box = Interval(column({0.0, 2.0}), column({1.0, 4.0})).zonotope();
    const Interval I = reset.evaluate(box)->interval();
    check(close(I.inf(), {1, -2}) && close(I.sup(), {2, -1}), b + ": the image of the box");
    // the input enters through B only
    const LinearReset input(Tensor::eye(2), Tensor({{1.0}, {0.0}}));
    const Zonotope U(column({0.0}), Tensor({{1.0}}));
    const Interval J = input.evaluate(box, U)->interval();
    check(close(J.inf(), {-1, 2}) && close(J.sup(), {2, 4}), b + ": with the input set");
    check(close(reset.evaluate(box, U)->interval().sup(), {2, -1}), b + ": no B, no input");
    // a bundle maps zonotope by zonotope; points column by column
    const ZonoBundle bundle({box, Zonotope(column({0.5, 3.0}), Tensor({{1.0}, {1.0}}))});
    const Set image = reset.evaluate(bundle);
    check(image.is<ZonoBundle>() && image.as<ZonoBundle>().parallel_sets() == 2,
          b + ": a bundle stays a bundle");
    const Tensor x = reset.evaluate(Tensor({{0.0, 1.0}, {2.0, 4.0}}));
    check(close(x, {1, 2, -1, -2}), b + ": two points");
    check(reset.pre_state_dim() == 2 && reset.post_state_dim() == 2 && reset.input_dim() == 0 &&
              input.input_dim() == 1,
          b + ": the dimensions");
}

/// Wrong shapes and classes are refused; with the checks off the image is the same.
void checks_and_their_switch() {
    // the constructor: A a matrix, B and c of its rows
    check(test::raises([] { LinearReset(Tensor::zeros({2})); }, Error::wrong_input_in_constructor),
          "constructor: A must be a matrix");
    check(test::raises([] { LinearReset(Tensor::eye(2), Tensor::eye(3)); },
                       Error::wrong_input_in_constructor),
          "constructor: B with the rows of A");
    check(test::raises([] { LinearReset(Tensor::eye(2), std::nullopt, column({1.0})); },
                       Error::wrong_input_in_constructor),
          "constructor: c with the rows of A");
    // evaluate: zonotopes and points of the pre-state dimension
    const LinearReset reset(Tensor::eye(2));
    check(test::raises([&] { reset.evaluate(Zonotope(column({1.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "evaluate: the dimension of the set");
    check(test::raises([&] { reset.evaluate(Interval(column({0.0, 0.0}), column({1.0, 1.0}))); },
                       Error::not_supported),
          "evaluate: a box is not a zonotope");
    check(test::raises([&] { reset.evaluate(Tensor({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "evaluate: points are columns");
    const Zonotope Z(column({1.0, 2.0}), Tensor::eye(2));
    const Tensor on = reset.evaluate(Z)->center();
    ChecksScope off(false);
    check(close(reset.evaluate(Z)->center(), on, 0), "evaluate: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { box_by_hand(b); });
    checks_and_their_switch();
    return test::finish("linear_reset evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
