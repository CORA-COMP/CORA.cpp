// test_con_zonotope_interval_cert - ConZonotope::interval_cert: the box from dual certificates
//
// A segment and a cut square are boxed exactly by the multipliers of the programs; a set without
// points is the empty box; any multipliers, even wrong ones, give a box that holds the set. On
// random sets the box holds the points, is as tight as CORA's exact ranges, and is never wider
// than the box of the zonotope (the stated properties).
//
// CORA.cpp only: tests interval_cert, CORALean's float replacement of the programs of interval
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/polytope/polytope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Multipliers of the given sizes, all the same value.
ConZonotope::Certificate aux_weights(int64_t n, std::size_t m, std::size_t p, double value) {
    const ConZonotope::CertWeights w{std::vector<double>(m, value), std::vector<double>(p, value),
                                     std::vector<double>(p, value)};
    return {std::vector<ConZonotope::CertWeights>(std::size_t(n), w),
            std::vector<ConZonotope::CertWeights>(std::size_t(n), w)};
}

/// The segment x1 + x2 = 1 in the unit square is boxed to [0, 1]^2; x1 = 3 has no point; the
/// square cut by x1 <= 0.25 is [-1, 0.25] x [-1, 1].
void by_hand(const std::string &b) {
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const Interval I = segment.interval_cert();
    check(close(I.inf(), {0, 0}, 1e-12) && close(I.sup(), {1, 1}, 1e-12), b + ": a segment");
    const ConZonotope empty(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({3.0}));
    check(empty.interval_cert().is_empty(), b + ": no point: the empty box");
    const ConZonotope square(Zonotope(column({0.0, 0.0}), Tensor::eye(2)));
    const Interval half =
        square.cut(Polytope(Tensor({{1.0, 0.0}}), column({0.25}))).interval_cert();
    check(close(half.inf(), {-1, -1}, 1e-12) && close(half.sup(), {0.25, 1}, 1e-12),
          b + ": a cut square");
    // a cut beyond the set has no point either
    check(square.cut(Polytope(Tensor({{1.0, 0.0}}), column({-3.0}))).interval_cert().is_empty(),
          b + ": a cut beyond the set");
}

/// Any multipliers give a box that holds the set: zero, large, negative (counted as zero) and
/// random ones; the zero ones give the box of the zonotope.
void arbitrary_weights(const std::string &b) {
    cora::Rng rng(5);
    const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
    const Tensor points = test::sample_points(cz, rng, 100);
    const Interval zono = Zonotope(cz.c(), cz.generators()).interval();
    for (const double value : {0.0, 1.0, 40.0, -3.0}) {
        const Interval I = cz.interval_cert(aux_weights(3, 0, 2, value));
        check(test::inside_box(points, I, 0.0), b + ": the box holds the points");
        // weights at or below zero are zero: the box of the zonotope
        if (value <= 0.0)
            check(close(I.inf(), zono.inf(), 1e-12) && close(I.sup(), zono.sup(), 1e-12),
                  b + ": no multipliers: the box of the zonotope");
    }
    // multipliers of the programs of a random direction, scrambled, still hold the points
    ConZonotope::Certificate w = aux_weights(3, 0, 2, 0.0);
    std::vector<double> noise(2 * 3 * 2);
    rng.normal(noise.data(), noise.size(), 2.0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 2; ++k) {
            w.up[i].zu[k] = noise[i * 4 + k];
            w.lo[i].zl[k] = noise[i * 4 + 2 + k];
        }
    check(test::inside_box(points, cz.interval_cert(w), 0.0), b + ": scrambled multipliers");
}

/// Stated properties: the points are held, the box is CORA's exact box up to rounding, inside the
/// box of the zonotope, and cert-tight is inside both of its parts.
void properties(const std::string &b) {
    cora::Rng rng(41);
    for (int trial = 0; trial < 6; ++trial) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Interval I = cz.interval_cert(), exact = cz.interval();
        const Tensor points = test::sample_points(cz, rng, 80);
        check(test::inside_box(points, I, 0.0), b + ": the box holds the points");
        check(close(I.inf(), exact.inf(), 1e-8) && close(I.sup(), exact.sup(), 1e-8),
              b + ": as tight as the exact ranges");
        const Interval zono = Zonotope(cz.c(), cz.generators()).interval();
        check(zono.inf().less_equal(I.inf() + 1e-12).min().item() > 0 &&
                  I.sup().less_equal(zono.sup() + 1e-12).min().item() > 0,
              b + ": inside the box of the zonotope");
    }
}

/// The cut of a random set by a polytope: the cert box holds the points of both and agrees with
/// the exact intersection's ranges; the cert-tight box lies inside the cert box and the tight box.
void cuts(const std::string &b) {
    cora::Rng rng(9);
    for (int trial = 0; trial < 4; ++trial) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1);
        const Polytope P(Tensor({{1.0, 0.5}, {-0.3, 1.0}}), A.c() + column({0.2, 0.1}));
        const ConZonotope::Cut K = A.cut(P);
        const Interval I = K.interval_cert(), exact = A.intersect(P).interval();
        check(!I.is_empty() && close(I.inf(), exact.inf(), 1e-8) &&
                  close(I.sup(), exact.sup(), 1e-8),
              b + ": a cut as tight as the exact intersection");
        const Interval tight = K.interval_tight(3), both = K.interval_cert_tight(3);
        check(I.inf().less_equal(both.inf()).min().item() > 0 &&
                  tight.inf().less_equal(both.inf()).min().item() > 0 &&
                  both.sup().less_equal(I.sup()).min().item() > 0 &&
                  both.sup().less_equal(tight.sup()).min().item() > 0,
              b + ": cert-tight inside both");
    }
}

/// Multipliers of the wrong sizes are refused, as is a batch; the same with checks off is
/// the same box.
void refusals() {
    const ConZonotope cz(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({0.5}));
    check(test::raises([&] { cz.interval_cert(aux_weights(2, 0, 1, 0.0)); },
                       Error::dimension_mismatch),
          "interval_cert: a multiplier per coordinate");
    check(test::raises([&] { cz.interval_cert(aux_weights(1, 0, 2, 0.0)); },
                       Error::dimension_mismatch),
          "interval_cert: a multiplier per row");
    const Interval on = cz.interval_cert();
    ChecksScope off(false);
    check(close(cz.interval_cert().sup(), on.sup(), 0), "interval_cert: the same with checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        by_hand(b);
        arbitrary_weights(b);
        properties(b);
        cuts(b);
    });
    refusals();
    return test::finish("con_zonotope interval_cert");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
