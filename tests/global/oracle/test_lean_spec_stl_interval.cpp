// test_lean_spec_stl_interval - the operations of StlInterval against the CORALean oracle
//
// Bit for bit at nearest on random intervals (bounded and unbounded, open and closed ends, points):
// the constructor, empty, isemptyobject, dim, infimum, supremum, interval, the closures, toLeft,
// toRight, and, plus and minus of an interval, a number and a box, minkDiff, contains of an
// interval and of a number, isequal and representsa. The tolerances are 0: CORALean decides
// withinTol on the reals, CORA.cpp in floats, which agree except at the edge of a tolerance.
//
// ORACLE: specification/@stlInterval/and_.m
// ORACLE: specification/@stlInterval/contains_.m
// ORACLE: specification/@stlInterval/dim.m
// ORACLE: specification/@stlInterval/empty.m
// ORACLE: specification/@stlInterval/infimum.m
// ORACLE: specification/@stlInterval/interval.m
// ORACLE: specification/@stlInterval/isemptyobject.m
// ORACLE: specification/@stlInterval/isequal.m
// ORACLE: specification/@stlInterval/leftClosure.m
// ORACLE: specification/@stlInterval/minkDiff.m
// ORACLE: specification/@stlInterval/minus.m
// ORACLE: specification/@stlInterval/plus.m
// ORACLE: specification/@stlInterval/representsa_.m
// ORACLE: specification/@stlInterval/rightClosure.m
// ORACLE: specification/@stlInterval/stlInterval.m
// ORACLE: specification/@stlInterval/supremum.m
// ORACLE: specification/@stlInterval/toLeft.m
// ORACLE: specification/@stlInterval/toRight.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/stl.h"
#include "cora/global/oracle/tensor.h"
#include "cora/specification/stl_interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// A whole number in [0, k).
int aux_pick(oracle::Random &rnd, int k) {
    return std::min(k - 1, static_cast<int>(rnd.uniform(0.0, static_cast<double>(k))));
}

/// A random interval: bounded or not, a point now and then, the ends open or closed at random.
StlInterval aux_interval(oracle::Random &rnd) {
    const double lb = aux_pick(rnd, 4) == 0 ? 0.0 : rnd.uniform(0.0, 2.0);
    const int shape = aux_pick(rnd, 6);
    const double ub = shape == 0 ? kInf : shape == 1 ? lb : lb + rnd.uniform(0.0, 2.0);
    const bool point = shape == 1;
    return StlInterval(lb, ub, point || aux_pick(rnd, 2) == 0, point || aux_pick(rnd, 2) == 0);
}

/// The same wire form, under the name `what`.
void aux_same(const std::string &what, const lean::Json &mine, const lean::Json &theirs) {
    const bool ok = mine.dump() == theirs.dump();
    if (!ok)
        std::cerr << "  " << what << ":\n    CORA.cpp " << mine.dump() << "\n    oracle   "
                  << theirs.dump() << "\n";
    test::check(ok, what + ": the oracle's interval");
}

/// The interval reply of `op` on I with further operands, against CORA.cpp's.
void aux_check(const char *op, const StlInterval &mine, const StlInterval &I,
               std::vector<lean::Operand> more = {}) {
    aux_same(op, lean::stl_json(mine),
             lean::ask_stl_interval(op, I, std::move(more)).get<lean::Json>("I"));
}

/// An end in the oracle's form: the bound ("inf", "nan" for the empty interval) and closed.
lean::Json aux_endpoint(const StlEndpoint &e) {
    lean::Json j = lean::Json::object();
    j.set("closed", e.closed);
    return j.set("value", std::isnan(e.value)   ? lean::Json("nan")
                          : std::isinf(e.value) ? lean::Json("inf")
                                                : lean::Json(lean::Tensor::encode(e.value)));
}

/// The end `which` (infimum, supremum) of the oracle in the same form.
lean::Json aux_their_endpoint(const char *op, const StlInterval &I) {
    const lean::Reply r = lean::ask_stl_interval(op, I);
    lean::Json j = lean::Json::object();
    j.set("closed", r.get<bool>("closed"));
    return j.set("value", r.get<lean::Json>("value"));
}

/// The interval operand `key`.
lean::Operand aux_operand(const char *key, const StlInterval &J) {
    return {key, lean::Wire(lean::stl_json(J), "")};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    constexpr int kTrials = 40;
    // the constructor, the empty interval and the queries of one interval
    oracle::run("stlInterval.stlInterval", [](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            // bounds in any order, a point now and then
            const double lb = rnd.uniform(0.0, 2.0);
            const double ub = aux_pick(rnd, 3) == 0 ? lb : rnd.uniform(0.0, 3.0);
            const bool lc = aux_pick(rnd, 2) == 0, rc = aux_pick(rnd, 2) == 0;
            aux_same("stlInterval", lean::stl_json(StlInterval(lb, ub, lc, rc)),
                     lean::ask("stlInterval.stlInterval",
                               {{"lb", lb}, {"ub", ub}, {"lc", lc}, {"rc", rc}})
                         .get<lean::Json>("I"));
        }
        // the empty interval of dimension 1
        aux_same("empty", lean::stl_json(StlInterval::empty(1)),
                 lean::ask("stlInterval.empty", {{"n", 1}}).get<lean::Json>("I"));
    });
    oracle::run("stlInterval.isemptyobject", [](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const StlInterval I = trial % 5 == 0 ? StlInterval() : aux_interval(rnd);
            oracle::check(
                "isemptyobject", I.is_empty_object(),
                lean::ask_stl_interval("stlInterval.isemptyobject", I).get<bool>("value"));
            oracle::check("dim", I.dim(),
                          lean::ask_stl_interval("stlInterval.dim", I).get<int64_t>("value"));
            aux_same("infimum", aux_endpoint(I.infimum()),
                     aux_their_endpoint("stlInterval.infimum", I));
            aux_same("supremum", aux_endpoint(I.supremum()),
                     aux_their_endpoint("stlInterval.supremum", I));
            aux_check("stlInterval.leftClosure", I.left_closure(), I);
            aux_check("stlInterval.rightClosure", I.right_closure(), I);
            if (I.is_empty_object()) continue;
            // the box of the bounds (not unbounded: CORALean's upper bound is 0 there)
            if (!std::isinf(I.upper()))
                oracle::check(
                    "interval", I.interval(),
                    lean::ask_stl_interval("stlInterval.interval", I).get<lean::Interval>("K"));
            // the half-lines beside the interval
            aux_check("stlInterval.toLeft", I.to_left(), I);
            aux_check("stlInterval.toRight", I.to_right(), I);
        }
    });
    // the operations of two intervals, and of an interval and a number or a box
    oracle::run("stlInterval.and", [](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const StlInterval I = aux_interval(rnd), J = aux_interval(rnd);
            const double c = rnd.uniform(0.0, 1.0);
            const double lo = rnd.uniform(0.0, 0.5), hi = lo + rnd.uniform(0.0, 1.0);
            const Interval K(Tensor::from_data({lo}, {1}), Tensor::from_data({hi}, {1}));
            const lean::Operand j = aux_operand("J", J), num{"c", c}, box{"K", lean::Interval(K)};
            aux_check("stlInterval.and", I.and_(J), I, {j});
            aux_check("stlInterval.plus", I.plus(J), I, {j});
            aux_check("stlInterval.plus", I.plus(c), I, {num});
            aux_check("stlInterval.plus", I.plus(K), I, {box});
            aux_check("stlInterval.minus", I.minus(J), I, {j});
            aux_check("stlInterval.minus", I.minus(c), I, {num});
            aux_check("stlInterval.minus", I.minus(K), I, {box});
            aux_check("stlInterval.minkDiff", I.minkowski_difference(J), I, {j});
        }
    });
    // the comparisons at tolerance 0
    oracle::run("stlInterval.contains", [](oracle::Random &rnd) {
        for (int trial = 0; trial < kTrials; ++trial) {
            const StlInterval I = aux_interval(rnd),
                              J = aux_pick(rnd, 3) == 0 ? I : aux_interval(rnd);
            const lean::Operand j = aux_operand("J", J), tol{"tol", 0.0};
            oracle::check(
                "contains", I.contains(J, 0.0),
                lean::ask_stl_interval("stlInterval.contains", I, {j, tol}).get<bool>("value"));
            // a number at the ends or anywhere
            for (const double t : {I.lower(), I.upper(), rnd.uniform(0.0, 4.0)}) {
                if (std::isinf(t)) continue;
                oracle::check("contains (number)", I.contains(t, 0.0),
                              lean::ask_stl_interval("stlInterval.contains", I, {{"t", t}, tol})
                                  .get<bool>("value"));
            }
            oracle::check(
                "isequal", I.isequal(J, 0.0),
                lean::ask_stl_interval("stlInterval.isequal", I, {j, tol}).get<bool>("value"));
            for (const char *type : {"emptySet", "stlInterval", "point", "origin"})
                oracle::check(
                    std::string("representsa ") + type, I.representsa(type, 0.0),
                    lean::ask_stl_interval("stlInterval.representsa", I, {{"type", type}, tol})
                        .get<bool>("value"));
        }
    });
    return test::finish("lean spec stl interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
