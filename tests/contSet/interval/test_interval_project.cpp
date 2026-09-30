// test_interval_project - interval project: the bounds of the selected dimensions

#include "contSet/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void project_keeps_the_selected_bounds(const std::string &b) {
    const Interval I(test::column({0, -1, 2}), test::column({3, 1, 5}));
    const std::unique_ptr<ContSet> P = I.project({1, 2});
    const Interval &Ip = static_cast<const Interval &>(*P);
    check(close(Ip.inf, std::vector<double>{-1, 2}) && close(Ip.sup, std::vector<double>{1, 5}),
          b + ": bounds of dimensions 1 and 2");
    const std::unique_ptr<ContSet> R = I.project({2, 0});
    check(close(static_cast<const Interval &>(*R).inf, std::vector<double>{2, 0}),
          b + ": the order is kept");
}

void project_rejects_unknown_dimensions(const std::string &b) {
    const Interval I(test::column({0, -1}), test::column({3, 1}));
    check(test::throws([&] { I.project({0, 2}); }), b + ": dimension 2 does not exist");
}

void project_rejects_an_interval_matrix(const std::string &b) {
    const Interval M(Tensor({{0, 0}, {0, 0}}), Tensor({{1, 1}, {1, 1}}));
    check(test::throws([&] { M.project({0}); }), b + ": a matrix is not projected");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        project_keeps_the_selected_bounds(b);
        project_rejects_unknown_dimensions(b);
        project_rejects_an_interval_matrix(b);
    });
    return test::finish("interval project");
}
