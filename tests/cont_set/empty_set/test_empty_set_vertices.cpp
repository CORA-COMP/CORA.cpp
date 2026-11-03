// test_empty_set_vertices - EmptySet vertices: none
//
// CORA: contSet/@emptySet/vertices_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/global/plot/plot.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_empty_set_has_no_polygon(const std::string &b) {
    check(EmptySet(2).vertices().empty(), b + ": no polygon");
}

/// Plotting needs project, interval, vertices and vertices_3d: the empty set draws nothing.
void the_empty_set_plots_as_nothing(const std::string &b) {
    for (const std::vector<int64_t> &dims : {std::vector<int64_t>{0}, {0, 1}, {0, 1, 2}}) {
        Figure fig;
        plot(fig, EmptySet(3), dims);
        std::size_t polygons = 0;
        for (const Figure::Layer &l : fig.layers()) polygons += l.polygons.size();
        check(polygons == 0, b + ": nothing drawn in " + std::to_string(dims.size()) + "-D");
        check(!fig.svg().empty() && !fig.tikz().empty(), b + ": the writers still work");
    }
}

/// A set of three dimensions has no polygon; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { EmptySet(3).vertices(); }, Error::not_supported),
          "vertices: three dimensions");
    ChecksScope off(false);
    check(EmptySet(2).vertices().empty(), "vertices: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        the_empty_set_has_no_polygon(b);
        the_empty_set_plots_as_nothing(b);
    });
    checks_and_their_switch();
    return test::finish("empty_set vertices");
}
