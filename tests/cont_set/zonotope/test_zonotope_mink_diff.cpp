// test_zonotope_mink_diff - zonotope mink_diff: the Minkowski difference by scaling the generators
//
// CORA: contSet/@zonotope/minkDiff.m
// PROPERTY: contSet/@zonotope/minkDiff.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

Zonotope square(double radius, std::vector<double> c = {0.0, 0.0}) {
    return Zonotope(column(c), Tensor({{radius, 0.0}, {0.0, radius}}));
}

/// Hand-computed in two dimensions (always the exact algorithm): aligned generators subtract; a
/// segment out of a square needs the program; a point and a box as subtrahend.
void mink_diff_known_2d(const std::string &b) {
    const Zonotope aligned = square(2.0, {1.0, 1.0}).mink_diff(square(1.0, {0.0, 0.5}));
    check(close(aligned.center(), std::vector<double>{1, 0.5}, 0) &&
              close(aligned.generators(), std::vector<double>{1, 0, 0, 1}, 0),
          b + ": aligned generators: [-1, 1]^2 less [-1, 1]^2 scaled");
    const Zonotope segment(column({0.0, 0.0}), Tensor({{0.5}, {0.5}}));
    const Zonotope half = square(1.0).mink_diff(segment);
    check(close(half.generators(), std::vector<double>{0.5, 0, 0, 0.5}, 1e-9),
          b + ": a segment out of a square");
    for (const char *method : {"approx", "inner", "outer", "outer:coarse", "exact"})
        check(close(square(1.0).mink_diff(segment, method).generators(), half.generators(), 1e-9),
              std::string(b + ": in two dimensions the method ") + method + " is the exact one");
    check(close(square(1.0).mink_diff(column({0.25, 0.0})).center(), std::vector<double>{-0.25, 0},
                0),
          b + ": a point shifts the center");
    check(close(square(1.0).mink_diff(Zonotope(column({0.5, 0.0}), Tensor::zeros({2, 0}))).center(),
                std::vector<double>{-0.5, 0}, 0),
          b + ": a zonotope without generators is a point");
}

/// Hand-computed in three dimensions: the cube [-2, 2]^3 less [-0.5, 0.5]^3 is [-1.5, 1.5]^3 with
/// every method; "exact" needs aligned generators there.
void mink_diff_known_3d(const std::string &b) {
    const Zonotope cube(column({0.0, 0.0, 0.0}), Tensor::eye(3) * 2.0);
    const Zonotope small(column({0.0, 0.0, 0.0}), Tensor::eye(3) * 0.5);
    for (const char *method : {"approx", "inner", "outer", "outer:coarse", "exact"}) {
        const Zonotope D = cube.mink_diff(small, method);
        check(D.generators().shape() == std::vector<int64_t>({3, 3}) &&
                  close(D.generators(), (Tensor::eye(3) * 1.5), 1e-9),
              std::string(b + ": the cube less a cube, ") + method);
    }
    const Zonotope segment(column({0.0, 0.0, 0.0}), Tensor({{0.5}, {0.0}, {0.0}}));
    check(close(cube.mink_diff(segment, "approx").generators(),
                std::vector<double>{1.5, 0, 0, 0, 2, 0, 0, 0, 2}, 1e-9),
          b + ": a segment along an axis shortens that generator (approx)");
    check(test::raises([&] { cube.mink_diff(segment, "exact"); }, Error::wrong_value),
          b + ": exact needs two dimensions or aligned generators");
}

/// Stated property: the inner results plus the subtrahend are in the minuend, the outer ones
/// contain the inner ones.
void mink_diff_property(const std::string &b) {
    cora::Rng rng(50);
    for (int k = 0; k < 10; ++k) {
        const Zonotope M = Zonotope::generate_random(rng, 2, std::nullopt, 4) * 2.0;
        const Zonotope S = Zonotope::generate_random(rng, 2, std::nullopt, 2) * 0.1 -
                           Zonotope::generate_random(rng, 2, std::nullopt, 2).center();
        const Zonotope inner = M.mink_diff(S, "inner");
        for (int j = 0; j < 40; ++j) {
            const std::vector<double> d = test::random_direction(rng, 2);
            check(test::support(inner, d) + test::support(S, d) <= test::support(M, d) + 1e-8,
                  b + ": inner + S leaves the minuend");
        }
    }
}

/// The refusals of CORA's unsupported options and the empty results, and the call with the checks
/// off.
void mink_diff_refuses() {
    const Zonotope cube(column({0.0, 0.0, 0.0}), Tensor::eye(3));
    const Zonotope small(column({0.0, 0.0, 0.0}), Tensor::eye(3) * 0.5);
    for (const char *method : {"inner:conZonotope", "inner:RaghuramanKoeln", "outer:scaling"})
        check(test::raises([&] { cube.mink_diff(small, method); }, Error::not_supported),
              std::string("mink_diff: ") + method);
    check(test::raises([&] { cube.mink_diff(small, "other"); }, Error::wrong_value),
          "mink_diff: unknown method");
    check(test::raises([&] { cube.mink_diff(square(1.0)); }, Error::dimension_mismatch),
          "mink_diff: dimensions");
    check(
        test::raises([&] { Zonotope::stack({cube, cube}).mink_diff(small); }, Error::not_supported),
        "mink_diff: a batch");
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(
        test::raises([&] { flat.mink_diff(Zonotope(column({0.0, 0.0}), Tensor({{0.1}, {-0.1}}))); },
                     Error::not_supported),
        "mink_diff: a flat minuend");
    check(test::raises([&] { flat.mink_diff(square(1.0)); }, Error::empty_set),
          "mink_diff: flat less full is empty");
    check(test::raises(
              [&] { square(1.0).mink_diff(Zonotope(column({0.0, 0.0}), Tensor({{3.0}, {0.0}}))); },
              Error::empty_set),
          "mink_diff: a subtrahend too large for the minuend");
    const Tensor on = square(1.0).mink_diff(square(0.5)).generators();
    ChecksScope off(false);
    check(close(square(1.0).mink_diff(square(0.5)).generators(), on, 0), "mink_diff: checks off");
}

/// A box as subtrahend is enclosed by a zonotope, with a warning.
void mink_diff_of_a_box(const std::string &b) {
    int warnings = 0;
    const WarningHandler previous =
        set_warning_handler([&](const std::string &, const std::string &) { ++warnings; });
    const Zonotope D = square(2.0).mink_diff(Interval(column({-0.5, -0.5}), column({0.5, 0.5})));
    set_warning_handler(previous);
    check(warnings == 1, b + ": a box as subtrahend warns once");
    check(close(D.generators(), std::vector<double>{1.5, 0, 0, 1.5}, 1e-9),
          b + ": [-2, 2]^2 less [-0.5, 0.5]^2");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        mink_diff_known_2d(b);
        mink_diff_known_3d(b);
        mink_diff_property(b);
        mink_diff_of_a_box(b);
    });
    mink_diff_refuses();
    return test::finish("zonotope mink_diff");
}
