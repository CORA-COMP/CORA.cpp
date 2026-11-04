// request - the one place that builds an oracle request and reads its response
//
// Syntax:     Reply r = ask("zonotope.supportFunc", {{"Z", Z}, {"dir", d}});
//             double v = r.get<double>("value");
// Operands:   Wire encodes a scalar, a boolean, a string, indices, a Tensor, an Interval, a
//             Zonotope or a list of them; its dtype (that of the lean value, "" for plain data)
//             sets the request's dtype, all lean operands must agree, none means lean::dtype()
// Results:    Reply::get<T>(key) decodes double, bool, int64 list, Tensor, Interval, Fullspace,
//             Zonotope, ConZonotope, MatZonotope, Zonotope list, or std::optional of one of them
//             (a JSON null is nothing); list(key) splits an array of objects into replies
// See also:   lean/oracle.h, lean/interval.h, lean/zonotope.h

#pragma once

#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/fullspace.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/mat_zonotope.h"
#include "cora/global/oracle/zonotope.h"

#include <initializer_list>
#include <optional>
#include <type_traits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// A request operand in wire form, built implicitly from the types an op takes.
struct Wire {
    Json json;
    std::string dtype;

    Wire(Json j, std::string dt) : json(std::move(j)), dtype(std::move(dt)) {}
    Wire(double x);
    Wire(bool b);
    Wire(const char *s);
    Wire(const std::string &s);
    template <class I, std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, bool>, int> = 0>
    Wire(I n) : json(static_cast<int64_t>(n)) {}
    Wire(const std::vector<int64_t> &indices);
    Wire(const std::vector<std::vector<int64_t>> &index_lists);
    Wire(const Tensor &T);
    Wire(const Interval &I);
    Wire(const Fullspace &F);
    Wire(const Zonotope &Z);
    Wire(const ConZonotope &cz);
    Wire(const MatZonotope &A);
    Wire(const std::vector<Tensor> &Ts);
    Wire(const std::vector<std::vector<Tensor>> &Tss);
    Wire(const std::vector<Zonotope> &Zs);
};

/// A plain JSON number, as the oracle reads its `tol` (the other scalars are exact texts).
Wire number(double x);

/// One member of a request: `{"Z", Z}`.
struct Operand {
    const char *key;
    Wire wire;
};

/// The parsed response of an op and the dtype its values are in.
class Reply {
  public:
    Reply(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    /// The member `key` as a T; throws if it is missing or not a T.
    template <class T>
    T get(const char *key) const;

    /// The elements of the array `key`, each a reply on its own: their members are read with get.
    std::vector<Reply> list(const char *key) const;

    /// The answer of a `contains` (a proof when true) or an `isIntersecting` (a proof of
    /// disjointness, so a proof when false) request.
    Certified proved_contains() const;
    Certified proved_disjoint() const;

  private:
    std::string dtype_;
    Json json_;
};

/// Sends {"op", "dtype", operands...} and returns the response; throws with the oracle's message.
Reply ask(const char *op, std::initializer_list<Operand> operands = {});

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
