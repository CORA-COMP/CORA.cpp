// test_interval_vertices - interval vertices: the four corners of a planar box

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void vertices_are_the_corners(const std::string &b) {
    const Interval I(test::column({-1, 2}), test::column({1, 5}));
    const std::vector<Polygon> V = I.vertices();
    const Polygon want = {{-1, 2}, {1, 2}, {1, 5}, {-1, 5}};
    check(V.size() == 1 && V[0] == want, b + ": lower left, lower right, upper right, upper left");
}

void vertices_need_a_planar_box(const std::string &b) {
    const Interval I(test::column({0, 0, 0}), test::column({1, 1, 1}));
    check(test::throws([&] { I.vertices(); }), b + ": three dimensions have no polygon");
}

/// A set of three dimensions is refused; with the checks off the polygon is the same.
void checks_and_their_switch() {
    const Interval S3 = Interval(test::column({0, -1, 0}), test::column({1, 1, 2}));
    check(test::raises([&] { S3.vertices(); }, Error::not_supported), "vertices: three dimensions");
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1}));
    const std::size_t on = S.vertices()[0].size();
    ChecksScope off(false);
    check(S.vertices()[0].size() == on, "vertices: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        vertices_are_the_corners(b);
        vertices_need_a_planar_box(b);
    });
    checks_and_their_switch();
    return test::finish("interval vertices");
}
