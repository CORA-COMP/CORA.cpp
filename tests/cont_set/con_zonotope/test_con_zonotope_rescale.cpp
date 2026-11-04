// test_con_zonotope_rescale - ConZonotope::rescale: every factor scaled to the range it takes
//
// CORA: contSet/@conZonotope/rescale.m
// PROPERTY: contSet/@conZonotope/rescale.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: x1 + x2 = 1.5 in the box [-1, 1]^2 leaves both factors in [0.5, 1]: the middle
/// 0.75 moves into the center and the offset, the radius 0.25 scales generators and constraint.
void rescale_known(const std::string &b) {
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({1.5}));
    for (const char *method : {"exact", "iter"}) {
        const ConZonotope r = a.rescale(method);
        check(close(r.c(), {0.75, 0.75}, 1e-9) && close(r.generators(), {0.25, 0, 0, 0.25}, 1e-9),
              b + ": the center and generators (" + method + ")");
        check(close(r.A(), {0.25, 0.25}, 1e-9) && close(r.b(), {0}, 1e-9),
              b + ": the constraint (" + method + ")");
    }
}

/// Stated property: the set is the same (its sampled points are in the rescaled set and the
/// boxes agree), and every factor now spans [-1, 1].
void rescale_property(const std::string &b) {
    cora::Rng rng(46);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        for (const char *method : {"exact", "iter"}) {
            const ConZonotope r = cz.rescale(method);
            check(r.contains(test::sample_points(cz, rng, 15)), b + ": same points (" + method + ")");
            check(close(r.interval().inf(), cz.interval().inf(), 1e-7) &&
                      close(r.interval().sup(), cz.interval().sup(), 1e-7),
                  b + ": same box (" + method + ")");
        }
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        rescale_known(b);
        rescale_property(b);
    });
    const ConZonotope plain(column({0.0}), Tensor({{1.0}}));
    const ConZonotope none(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({3.0}));
    check(test::raises([&] { plain.rescale(); }, Error::special_error), "rescale: no constraints");
    check(test::raises([&] { none.rescale(); }, Error::empty_set), "rescale: an empty set");
    check(test::raises([&] { plain.rescale("fast"); }, Error::wrong_value), "rescale: a method");
    ChecksScope off(false);
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({1.5}));
    check(close(a.rescale().b(), {0}, 1e-9), "rescale: the same with the checks off");
    return test::finish("con_zonotope rescale");
}
