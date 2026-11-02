// test_interval_project - interval project: the bounds of the selected dimensions

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void project_keeps_the_selected_bounds(const std::string &b) {
    const Interval I(test::column({0, -1, 2}), test::column({3, 1, 5}));
    const Set P = I.project({1, 2});
    const Interval &Ip = P.as<Interval>();
    check(close(Ip.inf(), std::vector<double>{-1, 2}) && close(Ip.sup(), std::vector<double>{1, 5}),
          b + ": bounds of dimensions 1 and 2");
    const Set R = I.project({2, 0});
    check(close(R.as<Interval>().inf(), std::vector<double>{2, 0}), b + ": the order is kept");
}

void project_rejects_unknown_dimensions(const std::string &b) {
    const Interval I(test::column({0, -1}), test::column({3, 1}));
    check(test::throws([&] { I.project({0, 2}); }), b + ": dimension 2 does not exist");
}

void project_rejects_an_interval_matrix(const std::string &b) {
    const Interval M(Tensor({{0, 0}, {0, 0}}), Tensor({{1, 1}, {1, 1}}));
    check(test::throws([&] { M.project({0}); }), b + ": a matrix is not projected");
}

/// No dimension or one outside the set is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { S.project({}); }, Error::wrong_value), "project: no dimension");
    check(test::raises([&] { S.project({2}); }, Error::wrong_value), "project: dimension 2 of 2");
    const Tensor on = S.project({1}).get().center();
    ChecksScope off(false);
    check(test::close(S.project({1}).get().center(), on, 0),
          "project: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        project_keeps_the_selected_bounds(b);
        project_rejects_unknown_dimensions(b);
        project_rejects_an_interval_matrix(b);
    });
    checks_and_their_switch();
    return test::finish("interval project");
}
