// test_linearSys_verifyTime - VerifyTime (the unverified times of the zonotope verification)
//
// Reference: MATLAB R2025b, CORA's verifyTime: T = verifyTime([0 10]); T = setdiff(T, op) for the
// operations below; timeUntilSwitch, contains, isIntersecting and unify on the results.

#include "contDynamics/linearSys/private/priv_verifyTime.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

bool same(const VerifyTime &T, const VerifyTime::Bounds &expected) {
    if (T.bounds.size() != expected.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (T.bounds[i] != expected[i]) return false;
    return true;
}

} // namespace

int main() {
    // setdiff: the pieces after each removal, as MATLAB returns them.
    VerifyTime T({{0, 10}});
    const double ops[13][2] = {{1, 2},   {3, 3.5}, {0, 0.5}, {2, 3},  {9, 12},  {5.5, 6}, {6, 7},
                               {7.5, 8}, {4, 4},   {3.5, 4.25}, {0.5, 1}, {12, 13}, {4.25, 5.5}};
    const VerifyTime::Bounds expected[13] = {
        {{0, 1}, {2, 10}},
        {{0, 1}, {2, 3}, {3.5, 10}},
        {{0.5, 1}, {2, 3}, {3.5, 10}},
        {{0.5, 1}, {3.5, 10}},
        {{0.5, 1}, {3.5, 9}},
        {{0.5, 1}, {3.5, 5.5}, {6, 9}},
        {{0.5, 1}, {3.5, 5.5}, {7, 9}},
        {{0.5, 1}, {3.5, 5.5}, {7, 7.5}, {8, 9}},
        {{0.5, 1}, {3.5, 4}, {4, 5.5}, {7, 7.5}, {8, 9}},
        {{0.5, 1}, {4.25, 5.5}, {7, 7.5}, {8, 9}},
        {{4.25, 5.5}, {7, 7.5}, {8, 9}},
        {{4.25, 5.5}, {7, 7.5}, {8, 9}},
        {{7, 7.5}, {8, 9}}};
    for (int i = 0; i < 13; ++i) {
        T.setdiff(ops[i][0], ops[i][1]);
        check(same(T, expected[i]), "setdiff " + std::to_string(i + 1));
    }

    // timeUntilSwitch and contains on [7 7.5; 8 9].
    const double ts[8] = {0, 0.75, 1.5, 2.5, 5, 7.2, 8.5, 9.2};
    const double until[8] = {7, 6.25, 5.5, 4.5, 2, 0.29999999999999982, 0.5, 0};
    const bool inside[8] = {false, false, false, false, false, true, true, false};
    const bool found[8] = {true, true, true, true, true, true, true, false};
    for (int i = 0; i < 8; ++i) {
        double tSwitch = 0;
        bool in = false;
        const bool ok = T.timeUntilSwitch(ts[i], tSwitch, in);
        check(ok == found[i], "a switch after " + std::to_string(ts[i]));
        if (ok) {
            check(test::close(tSwitch, until[i], 1e-14),
                  "time until the switch from " + std::to_string(ts[i]));
            check(in == inside[i], "inside after the switch from " + std::to_string(ts[i]));
        }
        check(T.contains(ts[i]) == (ts[i] == 7.2 || ts[i] == 8.5), "contains " + std::to_string(ts[i]));
    }

    // isIntersecting.
    const double iv[5][2] = {{0, 0.4}, {0.5, 0.6}, {2.5, 2.8}, {8, 9}, {8.2, 8.7}};
    const bool hit[5] = {false, false, false, true, true};
    for (int i = 0; i < 5; ++i)
        check(T.isIntersecting(iv[i][0], iv[i][1]) == hit[i], "isIntersecting " + std::to_string(i));

    // unify, and compact of touching intervals.
    const VerifyTime U = VerifyTime::unify(
        {VerifyTime({{0, 1}, {3, 4}}), VerifyTime({{0.5, 2}}), VerifyTime({{4, 5}, {7, 8}})});
    check(same(U, {{0, 2}, {3, 5}, {7, 8}}), "unify");
    check(same(VerifyTime({{0, 1}, {1, 2}, {3, 4}}).compact(), {{0, 2}, {3, 4}}), "compact");
    return test::finish("linearSys verifyTime");
}
