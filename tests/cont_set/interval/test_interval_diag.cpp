// test_interval_diag - interval diag: the diagonal matrix of a vector, the diagonal of a matrix
//
// CORA: contSet/@interval/diag.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void diag_of_a_vector(const std::string &b) {
    const Interval v(column({1.0, 2.0, 3.0}), column({4.0, 5.0, 6.0}));
    const Interval D = v.diag();
    check(D.inf().shape() == std::vector<int64_t>({3, 3}) &&
              close(D.inf(), std::vector<double>{1, 0, 0, 0, 2, 0, 0, 0, 3}) &&
              close(D.sup(), std::vector<double>{4, 0, 0, 0, 5, 0, 0, 0, 6}),
          b + ": the vector on the diagonal, [0, 0] elsewhere");
    const Interval U = Interval(column({1.0, 2.0}), column({3.0, 4.0})).diag(1);
    check(U.inf().shape() == std::vector<int64_t>({3, 3}) &&
              close(U.sup(), std::vector<double>{0, 3, 0, 0, 0, 4, 0, 0, 0}),
          b + ": k = 1 above the diagonal");
    const Interval L = Interval(column({1.0, 2.0}), column({3.0, 4.0})).diag(-2);
    check(L.inf().shape() == std::vector<int64_t>({4, 4}) &&
              close(L.inf().sum().data()[0], 3.0, 0) && close(L.sup().data()[8], 3.0, 0) &&
              close(L.sup().data()[13], 4.0, 0),
          b + ": k = -2 below the diagonal");
    const Tensor row = Tensor::from_data({1, 2}, {1, 2});
    check(close(Interval(row, row + 1.0).diag().sup(), std::vector<double>{2, 0, 0, 3}),
          b + ": a row is a vector too");
    const Interval one(column({7.0}), column({8.0}));
    check(close(one.diag().inf(), std::vector<double>{7}), b + ": a scalar");
    const Interval S = Interval::stack({v, v}).diag();
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 3}), b + ": a batch member by member");
}

void diag_of_a_matrix(const std::string &b) {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
    const Interval M(lo, lo + 1.0);
    check(close(M.diag().inf(), std::vector<double>{1, 5, 9}) &&
              M.diag().inf().shape() == std::vector<int64_t>({3, 1}),
          b + ": the main diagonal as a column");
    check(close(M.diag(1).sup(), std::vector<double>{3, 7}), b + ": k = 1");
    check(close(M.diag(-2).inf(), std::vector<double>{7}), b + ": k = -2");
    check(close(M.diag().diag().inf().sum().data()[0], 15.0),
          b + ": diag of diag keeps the diagonal");
}

/// An empty diagonal is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    const Interval M(lo, lo + 1.0);
    check(test::raises([&] { M.diag(2); }, Error::not_supported), "diag: k = 2 of a 2 x 2 matrix");
    check(test::raises([&] { M.diag(-2); }, Error::not_supported),
          "diag: k = -2 of a 2 x 2 matrix");
    const Tensor on = M.diag(1).sup();
    ChecksScope off(false);
    check(close(M.diag(1).sup(), on, 0), "diag: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        diag_of_a_vector(b);
        diag_of_a_matrix(b);
    });
    checks_and_their_switch();
    return test::finish("interval diag");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
