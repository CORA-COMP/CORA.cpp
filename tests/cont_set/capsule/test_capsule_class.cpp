// test_capsule_class - the constructors of Capsule: forms, batches and refusals
//
// CORA: contSet/@capsule/capsule.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        constructor_forms(b);
        batches(b);
    });
    refusals();
    return test::finish("capsule class");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
