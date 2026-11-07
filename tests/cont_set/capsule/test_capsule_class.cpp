// test_capsule_class - the constructors of Capsule: forms, batches and refusals
//
// CORA: contSet/@capsule/capsule.m
// CORA: contSet/@interval/capsule.m
// CORA: contSet/@zonotope/capsule.m
// PROPERTY: contSet/@zonotope/capsule.m
// PROPERTY: contSet/@interval/capsule.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   07-November-2026
// Reviewed:      ---

#include "capsule_testing.h"
#include "cora/cont_set/zonotope/zonotope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The forms of CORA's constructor: a center alone is a point, a generator gives a segment.
void constructor_forms(const std::string &b) {
    const Capsule point(column({1.0, 2.0}));
    check(point.dim() == 2 && close(point.c(), {1, 2}, 0) && close(point.generator(), {0, 0}, 0) &&
              close(point.r(), {0}, 0),
          b + ": a center alone is a point");
    const Capsule segment(column({1.0, 2.0}), column({3.0, 4.0}));
    check(close(segment.generator(), {3, 4}, 0) && close(segment.r(), {0}, 0),
          b + ": a segment has the radius 0");
    const Capsule full(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(close(full.r(), {0.5}, 0), b + ": the radius as a number");
    const Capsule tensor_radius(column({1.0, 2.0}), column({3.0, 4.0}),
                                Tensor::from_data({0.25}, {1, 1}));
    check(close(tensor_radius.r(), {0.25}, 0), b + ": the radius as a tensor");
    check(full.class_name() == "Capsule" && full.class_id() == ClassId::capsule,
          b + ": the class name and id");
    const Set via_set = full;
    check(via_set.is<Capsule>() && via_set.as<Capsule>().dim() == 2,
          b + ": a Set holds it as its class");
    check(!full.is_empty(), b + ": a capsule is not empty");
}

/// Leading axes are batches: one radius per capsule.
void batches(const std::string &b) {
    const Tensor c = Tensor::zeros({3, 2, 1}), g = Tensor::ones({3, 2, 1});
    const Capsule batch(c, g, Tensor::from_data({0.1, 0.2, 0.3}, {3, 1, 1}));
    check(batch.dim() == 2 && batch.r().shape() == Shape{3, 1, 1}, b + ": a batch of three");
    check(close(Capsule(c, g, 0.5).r(), {0.5, 0.5, 0.5}, 0), b + ": a number is every radius");
    check(test::raises([&] { Capsule(c, Tensor::ones({2, 2, 1})); },
                       Error::wrong_input_in_constructor),
          b + ": one batch shape for c and g");
    check(test::raises([&] { Capsule(c, g, Tensor::from_data({0.1, 0.2}, {2, 1, 1})); },
                       Error::wrong_value),
          b + ": one radius per capsule");
}

/// What the constructor refuses, and the same input with the checks off.
void refusals() {
    const Tensor c = column({1.0, 2.0}), g = column({3.0, 4.0});
    check(test::raises([&] { Capsule(Tensor({{1.0, 2.0}})); }, Error::wrong_value),
          "Capsule: the center is a column");
    check(test::raises([&] { Capsule(c, Tensor({{1.0, 2.0}})); }, Error::wrong_value),
          "Capsule: the generator is a column");
    check(test::raises([&] { Capsule(c, column({1.0})); }, Error::wrong_input_in_constructor),
          "Capsule: the generator has the dimension of the center");
    check(test::raises([&] { Capsule(c, g, -0.5); }, Error::wrong_value),
          "Capsule: the radius is nonnegative");
    check(test::raises([&] { Capsule(c, g, std::nan("")); }, Error::wrong_value),
          "Capsule: the radius is a number");
    check(test::raises([&] { Capsule(column({1.0, std::nan("")}), g); }, Error::wrong_value),
          "Capsule: finite entries");
    check(test::raises([&] { Capsule(Tensor::zeros({0, 1})); }, Error::wrong_value),
          "Capsule: a nonempty center");
    check(
        test::raises([&] { Capsule(c, g, Tensor::from_data({1, 2}, {1, 2})); }, Error::wrong_value),
        "Capsule: the radius is a scalar");
    // with the checks off the same valid call builds the same set
    const Capsule on(c, g, 0.5);
    ChecksScope off(false);
    const Capsule same(c, g, 0.5);
    check(close(same.r(), on.r(), 0) && close(same.c(), on.c(), 0),
          "Capsule: the same with the checks off");
}

/// The capsule of a box: the segment along the first widest edge, the ball over the other
/// half-widths.
void from_interval(const std::string &b) {
    const Interval box(column({0.0, 0.0, 0.0}), column({4.0, 2.0, 1.0}));
    const Capsule C(box);
    check(close(C.c(), {2, 1, 0.5}, 0) && close(C.generator(), {2, 0, 0}, 0),
          b + ": c and g of the widest edge");
    check(close(C.r(), {std::sqrt(1.25)}, 1e-14), b + ": r over the other half-widths");
    // the first of several widest edges is taken
    const Capsule tie(Interval(column({0.0, 0.0}), column({2.0, 2.0})));
    check(close(tie.generator(), {1, 0}, 0) && close(tie.r(), {1}, 0),
          b + ": the first widest edge");
    check(close(Capsule(Interval(column({1.0}), column({3.0}))).generator(), {1}, 0),
          b + ": one dimension");
    const Capsule batch(
        Interval(Tensor::zeros({2, 2, 1}), Tensor::from_data({1, 3, 2, 1}, {2, 2, 1})));
    check(close(batch.generator(), {0, 1.5, 1, 0}, 0), b + ": a batch");
}

/// Stated property: the capsule of a random box holds the box.
void holds_the_box(const std::string &b) {
    Rng rng(29);
    for (int k = 0; k < 5; ++k) {
        std::vector<double> c(3), r(3);
        rng.normal(c.data(), c.size(), 1.0);
        rng.uniform(r.data(), r.size(), 0.1, 2.0);
        const Interval box(column(c) - column(r), column(c) + column(r));
        check(Capsule(box).contains(box), b + ": the box is inside its capsule");
    }
}

/// Stated property: the capsule of a zonotope holds its points, for generators fewer and more than
/// n.
void from_zonotope(const std::string &b) {
    Rng rng(131);
    for (const int generators : {1, 2, 3, 7}) {
        std::vector<double> G(static_cast<std::size_t>(3 * generators));
        rng.normal(G.data(), G.size(), 1.0);
        const Zonotope Z(column({1.0, -1.0, 0.5}), Tensor::from_data(G, {3, generators}));
        const Capsule C(Z);
        check(C.contains(Z), b + ": the zonotope is inside the capsule of " +
                                 std::to_string(generators) + " generators");
        check(C.dim() == 3 && C.r().item() >= 0, b + ": a capsule of the dimension");
    }
    // a segment is a capsule of radius 0
    const Capsule segment(Zonotope(column({0.0, 0.0}), Tensor({{1.0}, {1.0}})));
    check(close(segment.r(), {0}, 1e-12), b + ": a zonotope with one generator has radius 0");
}

/// A batch of zonotopes and an interval matrix are refused.
void conversion_refusals() {
    const Zonotope batch(Tensor::zeros({2, 2, 1}), Tensor::zeros({2, 2, 1}));
    check(test::raises([&] { Capsule C(batch); }, Error::not_supported), "Capsule(Z): a batch");
    check(test::raises([] { Capsule C(Interval::empty(2)); }, Error::empty_set),
          "Capsule(I): empty");
    ChecksScope off(false);
    check(Capsule(Interval(column({0.0, 0.0}), column({1.0, 2.0}))).dim() == 2,
          "Capsule(I): works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        constructor_forms(b);
        batches(b);
        from_interval(b);
        holds_the_box(b);
        from_zonotope(b);
    });
    refusals();
    conversion_refusals();
    return test::finish("capsule class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
