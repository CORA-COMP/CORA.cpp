// test_interval_vertices_3d - interval vertices_3d: the eight corners of a box

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

bool has_corner(const std::vector<Point3> &corners, const Point3 &p) {
    for (const Point3 &q : corners)
        if (q == p) return true;
    return false;
}

void the_corners_of_a_box(const std::string &b) {
    const Interval I(test::column({0, -1, 2}), test::column({1, 3, 5}));
    const std::vector<std::vector<Point3>> V = I.vertices_3d();
    check(V.size() == 1 && V[0].size() == 8, b + ": eight corners of one box");
    for (double x : {0.0, 1.0})
        for (double y : {-1.0, 3.0})
            for (double z : {2.0, 5.0})
                check(has_corner(V[0], {x, y, z}), b + ": the corner is there");
}

void invalid_input_and_checks_off() {
    const Interval I2 = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { I2.vertices_3d(); }, Error::not_supported), "a plane is refused");
    const Interval I = Interval(test::column({0, 0, 0}), test::column({1, 1, 1}));
    ChecksScope off(false);
    check(I.vertices_3d()[0].size() == 8, "the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { the_corners_of_a_box(b); });
    invalid_input_and_checks_off();
    return test::finish("interval vertices_3d");
}
