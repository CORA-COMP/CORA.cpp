// test_zonotope_intersect_strip - zonotope intersect_strip: an enclosure of the intersection with
// strips
//
// CORA: contSet/@zonotope/intersectStrip.m
// PROPERTY: contSet/@zonotope/intersectStrip.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The unit square cut by the strip |x1 - 0.25| <= 0.5.
struct Strip {
    Zonotope Z{column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})};
    Tensor C = Tensor({{1.0, 0.0}}), phi = column({0.5}), y = column({0.25});
};

/// Hand-computed for the gain (1, 0), by "normGen" (0.8, 0) and "alamo-FRad" (the same for one
/// strip), "bravo" (the candidate of least volume) and "wang-FRad" (gain (0.5, 0)).
void strip_known(const std::string &b) {
    const Strip s;
    const Zonotope A = s.Z.intersect_strip(s.C, s.phi, s.y, Tensor::from_data({1.0, 0.0}, {2, 1}));
    check(close(A.center(), std::vector<double>{0.25, 0}, 1e-14), b + ": gain (1, 0): center");
    check(close(A.generators(), std::vector<double>{0, 0, 0.5, 0, 1, 0}, 1e-14),
          b + ": gain (1, 0): generators");

    const Zonotope N = s.Z.intersect_strip(s.C, s.phi, s.y);
    check(close(N.center(), std::vector<double>{0.2, 0}, 1e-14),
          b + ": normGen: center 0.8 * 0.25");
    check(close(N.generators(), std::vector<double>{0.2, 0, 0.4, 0, 1, 0}, 1e-14),
          b + ": normGen: generators");
    const Zonotope F = s.Z.intersect_strip(s.C, s.phi, s.y, "alamo-FRad");
    check(close(F.generators(), N.generators(), 1e-14) && close(F.center(), N.center(), 1e-14),
          b + ": alamo-FRad for one strip");
    const Zonotope R = s.Z.intersect_strip(s.C, s.phi, s.y, "bravo");
    check(close(R.center(), std::vector<double>{0.25, 0}, 1e-14), b + ": bravo: center");
    check(close(R.generators(), std::vector<double>{0.5, 0, 0, 1}, 1e-14),
          b + ": bravo: the candidate of volume 0.25");
    const Zonotope::StripObserver obs{Tensor::zeros({2, 1}), column({1.0}), Tensor::eye(2), s.C};
    const Zonotope W = s.Z.intersect_strip(s.C, s.phi, s.y, obs);
    check(close(W.center(), std::vector<double>{0.125, 0}, 1e-14), b + ": wang-FRad: center");
    check(close(W.generators(), std::vector<double>{0.5, 0, 0.25, 0, 1, 0}, 1e-14),
          b + ": wang-FRad: generators");
}

/// Stated property: for every gain the points of Z inside the strips are in the result.
void strip_encloses_the_intersection(const std::string &b) {
    cora::Rng rng(33);
    const Zonotope Z = Zonotope::generate_random(3, 5, rng);
    const Tensor C = Tensor({{1.0, 0.5, 0.0}, {0.0, -1.0, 2.0}});
    const Tensor phi = column({0.8, 1.5}), y = C.matmul(Z.center()) + column({0.3, -0.2});
    const Tensor points = Z.rand_point(3000, rng);
    // the points of Z inside all strips of (C, phi, y)
    const auto inside = [&](const Tensor &C_, const Tensor &phi_, const Tensor &y_) {
        const Tensor excess = (C_.matmul(points) - y_).abs() - phi_;
        return excess.max(0, true).less_equal(excess.max(0, true).zeros_like()).data();
    };
    const auto check_all = [&](const Zonotope &result, const std::vector<double> &keep,
                               const std::string &what) {
        int count = 0;
        for (int k = 0; k < 3000; ++k) {
            if (keep[k] == 0) continue;
            ++count;
            check(result.contains(points.slice(-1, k, k + 1), "exact", 1e-9),
                  b + ": " + what + ": a point of the intersection leaves the result");
        }
        check(count > 100, b + ": " + what + ": the strips keep part of the zonotope");
    };
    const std::vector<double> both = inside(C, phi, y);
    check_all(Z.intersect_strip(C, phi, y), both, "normGen");
    check_all(Z.intersect_strip(C, phi, y, Tensor::zeros({3, 2})), both, "gain 0");
    check_all(Z.intersect_strip(C, phi, y, "bravo"),
              inside(C.slice(0, 0, 1), phi.slice(0, 0, 1), y.slice(0, 0, 1)), "bravo");
    const Tensor C1 = C.slice(0, 0, 1), phi1 = phi.slice(0, 0, 1), y1 = y.slice(0, 0, 1);
    check_all(Z.intersect_strip(C1, phi1, y1, "alamo-FRad"), inside(C1, phi1, y1), "alamo-FRad");
}

/// The refusals of CORA's unsupported methods and wrong inputs, and the call with the checks off.
void strip_refuses() {
    const Strip s;
    for (const char *method : {"svd", "radius", "alamo-volume"})
        check(test::raises([&] { s.Z.intersect_strip(s.C, s.phi, s.y, method); },
                           Error::not_supported),
              std::string("intersect_strip: ") + method);
    check(test::raises([&] { s.Z.intersect_strip(s.C, s.phi, s.y, "other"); }, Error::wrong_value),
          "intersect_strip: unknown method");
    check(test::raises([&] { s.Z.intersect_strip(s.C, s.phi, s.y, "wang-FRad"); },
                       Error::wrong_value),
          "intersect_strip: wang without matrices");
    check(test::raises([&] { s.Z.intersect_strip(Tensor({{1.0, 0.0, 0.0}}), s.phi, s.y); },
                       Error::dimension_mismatch),
          "intersect_strip: C columns");
    check(test::raises([&] { s.Z.intersect_strip(s.C, column({0.5, 0.5}), s.y); },
                       Error::dimension_mismatch),
          "intersect_strip: phi rows");
    check(test::raises([&] { s.Z.intersect_strip(s.C, s.phi, column({0.5, 0.5})); },
                       Error::dimension_mismatch),
          "intersect_strip: y rows");
    check(test::raises([&] { s.Z.intersect_strip(s.C, s.phi, s.y, Tensor::zeros({2, 2})); },
                       Error::dimension_mismatch),
          "intersect_strip: gain columns");
    const Tensor C2 = Tensor({{1.0, 0.0}, {0.0, 1.0}});
    check(
        test::raises(
            [&] { s.Z.intersect_strip(C2, column({1.0, 1.0}), column({0.0, 0.0}), "alamo-FRad"); },
            Error::special_error),
        "intersect_strip: alamo-FRad with two strips");
    check(
        test::raises([&] { Zonotope::stack({s.Z, s.Z}).intersect_strip(s.C, s.phi, s.y, "bravo"); },
                     Error::not_supported),
        "intersect_strip: bravo on a batch");
    const Tensor on = s.Z.intersect_strip(s.C, s.phi, s.y).generators();
    ChecksScope off(false);
    check(close(s.Z.intersect_strip(s.C, s.phi, s.y).generators(), on, 0),
          "intersect_strip: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        strip_known(b);
        strip_encloses_the_intersection(b);
    });
    strip_refuses();
    return test::finish("zonotope intersect_strip");
}
