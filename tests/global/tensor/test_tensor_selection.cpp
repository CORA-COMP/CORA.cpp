// test_tensor_selection - select_cols, cat_rows and max_last, on every backend
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void select_cols_takes_columns_in_order(const std::string &b) {
    const Tensor a = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    const Tensor s = a.take({2, 0}, -1);
    check(s.shape() == std::vector<int64_t>({2, 2}), b + ": shape");
    check(test::close(s, std::vector<double>{3.0, 1.0, 6.0, 4.0}), b + ": columns 2 and 0");
    check(test::close(a.take({1}, -1), std::vector<double>{2.0, 5.0}), b + ": one column");
}

void cat_rows_stacks_below(const std::string &b) {
    const Tensor a = Tensor::from_data({1.0, 2.0}, {1, 2}),
                 c = Tensor::from_data({3.0, 4.0, 5.0, 6.0}, {2, 2});
    const Tensor r = Tensor::concat({a, c}, -2);
    check(r.shape() == std::vector<int64_t>({3, 2}), b + ": shape");
    check(test::close(r, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), b + ": rows in order");
}

void max_last_takes_the_row_maximum(const std::string &b) {
    const Tensor a = Tensor::from_data({1.0, 7.0, 3.0, -4.0, -5.0, -6.0}, {2, 3});
    const Tensor m = a.max(-1, true);
    check(m.shape() == std::vector<int64_t>({2, 1}), b + ": shape");
    check(test::close(m, std::vector<double>{7.0, -4.0}), b + ": the largest of every row");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        select_cols_takes_columns_in_order(b);
        cat_rows_stacks_below(b);
        max_last_takes_the_row_maximum(b);
    });
    return test::finish("tensor selection");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
