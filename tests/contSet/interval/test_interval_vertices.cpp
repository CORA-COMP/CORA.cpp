// test_interval_vertices - interval vertices: the four corners of a planar box

#include "contSet/interval/interval.h"
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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        vertices_are_the_corners(b);
        vertices_need_a_planar_box(b);
    });
    return test::finish("interval vertices");
}
