// test_con_zonotope_class - the constructors of ConZonotope: forms, conversions and refusals
//
// CORA: contSet/@conZonotope/conZonotope.m
// CORA: contSet/@interval/conZonotope.m
// CORA: contSet/@zonotope/conZonotope.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The forms of CORA's constructor and the conversions from a zonotope and a box.
void constructor_forms(const std::string &b) {
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(plain.dim() == 2 && plain.A().shape() == Shape{0, 2} && plain.b().shape() == Shape{0, 1},
          b + ": without constraints, A has no rows");
    check(plain.class_name() == "ConZonotope", b + ": the class name");
    const ConZonotope constrained(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}),
                                  Tensor({{1.0, 1.0}}), column({0.5}));
    check(close(constrained.A(), {1, 1}, 0) && close(constrained.b(), {0.5}, 0), b + ": A and b");
    // the legacy form [c G]
    const ConZonotope legacy = ConZonotope::from_matrix(Tensor({{1.0, 1.0, 0.0}, {2.0, 0.0, 3.0}}),
                                                        Tensor({{1.0, 1.0}}), column({0.5}));
    check(close(legacy.generators(), constrained.generators(), 0) &&
              close(legacy.c(), constrained.c(), 0),
          b + ": from_matrix splits the first column");
    check(ConZonotope::from_matrix(Tensor({{1.0}, {2.0}})).generators().shape() == Shape{2, 0},
          b + ": a center only: no generators");
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(close(ConZonotope(Z).generators(), Z.generators(), 0), b + ": from a zonotope");
    const Interval I(column({0.0, 1.0}), column({2.0, 5.0}));
    check(close(ConZonotope(I).c(), {1, 3}, 0), b + ": from a box: its center");
    check(close(ConZonotope(I).generators(), {1, 0, 0, 2}, 0), b + ": from a box: its radius");
    const Set via_set = ConZonotope::empty(2);
    check(via_set.is<ConZonotope>() && via_set.class_id() == ClassId::con_zonotope,
          b + ": a Set holds it as its class");
}

/// What the constructor refuses, and the same input with the checks off.
void refusals() {
    const Tensor G({{1.0, 0.0}, {0.0, 3.0}});
    check(test::raises([&] { ConZonotope(Tensor({{1.0, 2.0}}), G); },
                       Error::wrong_input_in_constructor),
          "ConZonotope: the center is a column");
    check(test::raises([&] { ConZonotope(column({1.0}), G); }, Error::wrong_input_in_constructor),
          "ConZonotope: the rows of G are the dimension");
    check(test::raises([&] { ConZonotope(column({1.0, 2.0}), G, Tensor({{1.0}}), column({1.0})); },
                       Error::wrong_input_in_constructor),
          "ConZonotope: A has a column per generator");
    check(test::raises(
              [&] { ConZonotope(column({1.0, 2.0}), G, Tensor({{1.0, 1.0}}), column({1.0, 2.0})); },
              Error::wrong_input_in_constructor),
          "ConZonotope: b has a row per row of A");
    check(test::raises([&] { ConZonotope(column({1.0, std::nan("")}), G); },
                       Error::wrong_input_in_constructor),
          "ConZonotope: finite entries");
    check(test::raises([&] { ConZonotope::from_matrix(Tensor({{1.0}}), Tensor({{1.0}})); },
                       Error::wrong_input_in_constructor),
          "from_matrix: A and b together");
    check(test::raises([&] { ConZonotope::from_matrix(Tensor::zeros({2, 0})); },
                       Error::wrong_input_in_constructor),
          "from_matrix: at least the center");
    // with the checks off the same valid call builds the same set
    const ConZonotope on(column({1.0, 2.0}), G, Tensor({{1.0, 1.0}}), column({0.5}));
    ChecksScope off(false);
    const ConZonotope same(column({1.0, 2.0}), G, Tensor({{1.0, 1.0}}), column({0.5}));
    check(close(same.A(), on.A(), 0), "ConZonotope: the same with the checks off");
}

/// The batch: sets with a leading axis share one batch shape.
void batches(const std::string &b) {
    const Tensor c = Tensor::zeros({3, 2, 1}), G = Tensor::ones({3, 2, 4});
    const ConZonotope cz(c, G, Tensor::zeros({3, 1, 4}), Tensor::zeros({3, 1, 1}));
    check(cz.dim() == 2 && cz.A().shape() == Shape{3, 1, 4}, b + ": a batch of three sets");
    check(test::raises([&] { ConZonotope(c, Tensor::ones({2, 2, 4})); },
                       Error::wrong_input_in_constructor),
          b + ": one batch shape");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        constructor_forms(b);
        batches(b);
    });
    refusals();
    return test::finish("con_zonotope class");
}
