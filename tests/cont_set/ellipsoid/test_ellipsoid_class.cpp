// test_ellipsoid_class - the constructors of Ellipsoid: forms, defaults, checks and refusals
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA: contSet/@interval/ellipsoid.m
// CORA: contSet/@zonotope/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The forms of CORA's constructor: with and without a center, the tolerance, the batch.
void constructor_forms(const std::string &b) {
    const Tensor Q({{4.0, 0.0}, {0.0, 1.0}});
    const Ellipsoid centered(Q, column({1.0, 2.0}));
    check(close(centered.Q(), Q, 0) && close(centered.q(), {1, 2}, 0), b + ": Q and q");
    check(centered.dim() == 2 && centered.class_name() == "Ellipsoid", b + ": dimension and name");
    const Ellipsoid origin(Q);
    check(close(origin.q(), {0, 0}, 0) && origin.q().shape() == Shape{2, 1},
          b + ": the center defaults to the origin");
    check(Ellipsoid(Q, 1e-6).tol() == 1e-6, b + ": a tolerance alone");
    check(Ellipsoid(Q, column({0.0, 0.0}), 1e-6).tol() == 1e-6, b + ": the tolerance is kept");
    check(Ellipsoid(Q).tol() == default_tol(Q.dtype()), b + ": the default is the repository one");
    const Set via_set = centered;
    check(via_set.is<Ellipsoid>() && via_set.class_id() == ClassId::ellipsoid,
          b + ": a Set holds it as its class");
    const Ellipsoid batch(Tensor::zeros({3, 2, 2}), Tensor::zeros({3, 2, 1}));
    check(batch.dim() == 2 && batch.Q().shape() == Shape{3, 2, 2}, b + ": a batch of three");
    // asymmetry and a negative eigenvalue within tol are accepted, as CORA does
    check(!test::throws([&] { Ellipsoid(Tensor({{1.0, 1e-8}, {0.0, 1.0}}), 1e-6); }),
          b + ": an asymmetry below tol");
    check(!test::throws([&] { Ellipsoid(Tensor({{1.0, 0.0}, {0.0, -1e-8}}), 1e-6); }),
          b + ": an eigenvalue above -tol");
}

/// What the constructor refuses, and the same input with the checks off.
void refusals() {
    const Tensor Q({{4.0, 0.0}, {0.0, 1.0}});
    const auto refused = [](const Tensor &q, const Tensor &Qm) {
        return test::raises([&] { Ellipsoid(Qm, q); }, Error::wrong_input_in_constructor);
    };
    check(refused(column({0.0, 0.0}), Tensor({{1.0, 2.0, 3.0}, {0.0, 1.0, 1.0}})),
          "Ellipsoid: Q is square");
    check(refused(column({0.0}), Q), "Ellipsoid: q has a row per row of Q");
    check(refused(Tensor({{0.0, 0.0}}), Q), "Ellipsoid: q is a column");
    check(refused(column({0.0, 0.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}})), "Ellipsoid: Q symmetric");
    check(refused(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, -1.0}})), "Ellipsoid: Q psd");
    check(refused(column({0.0, std::nan("")}), Q), "Ellipsoid: q finite");
    check(refused(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, INFINITY}})),
          "Ellipsoid: Q finite");
    check(refused(column({0.0}), Tensor::zeros({0, 0})), "Ellipsoid: empty Q needs an empty q");
    check(test::raises([&] { Ellipsoid(Q, -1.0); }, Error::wrong_value),
          "Ellipsoid: the tolerance is nonnegative");
    check(test::raises([&] { Ellipsoid(Tensor::zeros({3, 2, 2}), Tensor::zeros({2, 2, 1})); },
                       Error::wrong_input_in_constructor),
          "Ellipsoid: one batch shape");
    // with the checks off the same valid call builds the same set, and an asymmetric Q is kept
    const Ellipsoid on(Q, column({1.0, 2.0}));
    ChecksScope off(false);
    const Ellipsoid same(Q, column({1.0, 2.0}));
    check(close(same.Q(), on.Q(), 0) && close(same.q(), on.q(), 0),
          "Ellipsoid: the same with the checks off");
    const Ellipsoid kept(Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    check(close(kept.Q(), {1, 0.5, 0, 1}, 0), "Ellipsoid: no check, no refusal");
}

/// The ellipsoid of a box: n times its diagonal shape; radii below 5e-3 are widened to it.
void from_interval(const std::string &b) {
    const Interval box(column({0.0, 0.0}), column({2.0, 1.0}));
    const Ellipsoid E(box);
    check(close(E.Q(), {2, 0, 0, 0.5}, 1e-14) && close(E.q(), {1, 0.5}, 0), b + ": n diag(rad^2)");
    const Ellipsoid thin(Interval(column({0.0, 0.0}), column({2.0, 0.002})));
    check(close(thin.Q(), {2, 0, 0, 2 * 5e-3 * 5e-3}, 1e-14),
          b + ": a thin edge is widened to 5e-3");
    // the ellipsoid holds the box, by its corners
    check(E.contains(box), b + ": holds the box");
    check(Ellipsoid(Interval(column({1.0}), column({3.0}))).Q().item() == 1.0,
          b + ": one dimension");
}

/// The ellipsoid of n independent generators is n G G' (outer) or G G' (inner); a point has Q = 0.
void from_zonotope(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(Ellipsoid(Z).Q(), {2, 0, 0, 8}, 1e-14) && close(Ellipsoid(Z).q(), {1, 2}, 0),
          b + ": outer");
    check(close(Ellipsoid(Z, "inner:norm").Q(), {1, 0, 0, 4}, 1e-14), b + ": inner");
    check(Ellipsoid(Z, "outer:exact").contains(Z), b + ": the outer ellipsoid holds the zonotope");
    check(
        close(Ellipsoid(Zonotope(column({1.0, 2.0}), Tensor::zeros({2, 0}))).Q(), {0, 0, 0, 0}, 0),
        b + ": a point");
}

/// Boxes with a flat edge, the inner mode, zonotopes with more generators and a batch are refused.
void conversion_refusals() {
    const Interval flat(column({0.0, 0.0}), column({1.0, 0.0}));
    check(test::raises([&] { Ellipsoid E(flat); }, Error::not_supported),
          "Ellipsoid(I): a flat edge");
    check(test::raises([&] { Ellipsoid E(flat, "inner"); }, Error::not_supported),
          "Ellipsoid(I): inner");
    check(test::raises([&] { Ellipsoid E(flat, "middle"); }, Error::wrong_value),
          "Ellipsoid(I): mode");
    const Zonotope many(column({0.0, 0.0}), Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    check(test::raises([&] { Ellipsoid E(many); }, Error::not_supported),
          "Ellipsoid(Z): more generators");
    check(test::raises([&] { Ellipsoid E(many, "outer"); }, Error::wrong_value),
          "Ellipsoid(Z): mode");
    ChecksScope off(false);
    check(Ellipsoid(Interval(column({0.0, 0.0}), column({1.0, 2.0}))).dim() == 2,
          "Ellipsoid(I): works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        constructor_forms(b);
        from_interval(b);
        from_zonotope(b);
    });
    refusals();
    conversion_refusals();
    return test::finish("ellipsoid class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
