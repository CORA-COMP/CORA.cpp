// test_ellipsoid_interval - Ellipsoid::interval: the box [q - r, q + r], r = sqrt(diag Q)
//
// CORA: contSet/@ellipsoid/interval.m
// PROPERTY: contSet/@ellipsoid/interval.m

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the radii are the roots of the diagonal, off-diagonal entries do not matter.
void interval_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 1.0}, {1.0, 1.0}}), column({1.0, 2.0}));
    const Interval I = E.interval();
    check(close(I.inf(), {-1, 1}, 0) && close(I.sup(), {3, 3}, 0), b + ": q -+ sqrt(diag Q)");
    const Interval point = Ellipsoid(Tensor::zeros({2, 2}), column({1.0, 2.0})).interval();
    check(close(point.inf(), {1, 2}, 0) && close(point.sup(), {1, 2}, 0), b + ": a point");
    check(Ellipsoid::empty(2).interval().is_empty(), b + ": the empty ellipsoid");
    const Ellipsoid batch(Tensor::stack({Tensor({{4.0}}), Tensor({{9.0}})}, 0),
                          Tensor::stack({column({0.0}), column({1.0})}, 0));
    check(close(batch.interval().sup(), Tensor::stack({column({2.0}), column({4.0})}, 0), 0),
          b + ": a batch");
}

/// Stated property: the box is the smallest one that holds the ellipsoid. Points of the boundary
/// are inside it, and the points where it touches (q +- Q e_i / r_i) are on the boundary.
void interval_property(const std::string &b) {
    cora::Rng rng(17);
    for (int k = 0; k < 5; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k == 4 ? 2 : -1);
        const Interval I = E.interval();
        const Tensor P = test::ellipsoid_points(E, rng, 50, true);
        const std::vector<double> lo = I.inf().data(), hi = I.sup().data(), x = P.data();
        bool inside = true;
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t j = 0; j < 50; ++j)
                inside = inside && x[i * 50 + j] >= lo[i] - 1e-12 && x[i * 50 + j] <= hi[i] + 1e-12;
        check(inside, b + ": boundary points are in the box");
        const Tensor r = (I.sup() - I.inf()) * 0.5;
        for (int i = 0; i < 3; ++i) {
            // the touching point q + Q e_i / r_i
            const Tensor touch = E.q() + E.Q().slice(1, i, i + 1) / r.data()[i];
            check(test::finite_near(E.ellipsoid_norm(touch - E.q()), 1.0) &&
                      close(touch.slice(0, i, i + 1), I.sup().slice(0, i, i + 1), 1e-12),
                  b + ": the box touches the ellipsoid in coordinate i");
        }
    }
}

/// No input to get wrong; with the checks off the box is the same.
void switch_off() {
    const Ellipsoid E(Tensor({{4.0}}), column({1.0}));
    const Interval on = E.interval();
    ChecksScope off(false);
    check(close(E.interval().sup(), on.sup(), 0), "interval: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        interval_known(b);
        interval_property(b);
    });
    switch_off();
    return test::finish("ellipsoid interval");
}
