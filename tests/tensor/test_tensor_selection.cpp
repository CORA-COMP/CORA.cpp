// test_tensor_selection - selectCols, catRows and maxLast, on every backend

#include "testing.h"

using namespace cora;
using test::check;

namespace {

void select_cols_takes_columns_in_order(const std::string &b) {
    const Tensor a = Tensor::fromData({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    const Tensor s = a.selectCols({2, 0});
    check(s.shape() == std::vector<int64_t>({2, 2}), b + ": shape");
    check(test::close(s, std::vector<double>{3.0, 1.0, 6.0, 4.0}), b + ": columns 2 and 0");
    check(test::close(a.selectCols({1}), std::vector<double>{2.0, 5.0}), b + ": one column");
}

void cat_rows_stacks_below(const std::string &b) {
    const Tensor a = Tensor::fromData({1.0, 2.0}, {1, 2}), c = Tensor::fromData({3.0, 4.0, 5.0, 6.0}, {2, 2});
    const Tensor r = Tensor::catRows({a, c});
    check(r.shape() == std::vector<int64_t>({3, 2}), b + ": shape");
    check(test::close(r, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}), b + ": rows in order");
}

void max_last_takes_the_row_maximum(const std::string &b) {
    const Tensor a = Tensor::fromData({1.0, 7.0, 3.0, -4.0, -5.0, -6.0}, {2, 3});
    const Tensor m = a.maxLast();
    check(m.shape() == std::vector<int64_t>({2, 1}), b + ": shape");
    check(test::close(m, std::vector<double>{7.0, -4.0}), b + ": the largest of every row");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        select_cols_takes_columns_in_order(b);
        cat_rows_stacks_below(b);
        max_last_takes_the_row_maximum(b);
    });
    return test::finish("tensor selection");
}
