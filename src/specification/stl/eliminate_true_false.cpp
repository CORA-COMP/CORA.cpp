// eliminate_true_false - the formula without true and false below the operators
//
// Syntax:   res = phi.eliminate_true_false();
// Inputs:   none
// Outputs:  res - the formula; true and false only remain as the whole formula
// See also: stl.h
//
// CORA: specification/@stl/eliminateTrueFalse.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::eliminate_true_false() const {
    detail::Entry entry;
    // an identified sub-formula stays as it is
    if (id() != 0) return *this;
    const auto is = [](const Stl &p, StlKind k) { return p.kind() == k; };
    switch (kind()) {
    case StlKind::and_: {
        const Stl l = lhs().eliminate_true_false(), r = rhs().eliminate_true_false();
        if (is(l, StlKind::false_) || is(r, StlKind::false_)) return Stl(false);
        if (is(l, StlKind::true_)) return r;
        return is(r, StlKind::true_) ? l : l.and_(r);
    }
    case StlKind::or_: {
        const Stl l = lhs().eliminate_true_false(), r = rhs().eliminate_true_false();
        if (is(l, StlKind::true_) || is(r, StlKind::true_)) return Stl(true);
        if (is(l, StlKind::false_)) return r;
        return is(r, StlKind::false_) ? l : l.or_(r);
    }
    case StlKind::not_: {
        const Stl in = lhs().eliminate_true_false();
        if (is(in, StlKind::false_) || is(in, StlKind::true_)) return Stl(is(in, StlKind::false_));
        return in.not_();
    }
    // a constant operand decides next, finally and globally
    case StlKind::next:
    case StlKind::finally:
    case StlKind::globally: {
        const Stl in = lhs().eliminate_true_false();
        if (is(in, StlKind::false_) || is(in, StlKind::true_)) return in;
        if (kind() == StlKind::next) return in.next(from());
        return kind() == StlKind::finally ? in.finally(interval()) : in.globally(interval());
    }
    case StlKind::until: {
        const Stl l = lhs().eliminate_true_false(), r = rhs().eliminate_true_false();
        if (is(l, StlKind::true_)) return r.finally(interval());
        // CORA rewrites phi U true to true; CORALean keeps the node (see docs/cora-differences.md)
        if (is(r, StlKind::true_)) return l.until(r, interval());
        return is(r, StlKind::false_) ? Stl(false) : l.until(r, interval());
    }
    case StlKind::release: {
        const Stl l = lhs().eliminate_true_false(), r = rhs().eliminate_true_false();
        if (is(l, StlKind::false_)) return r.globally(interval());
        if (is(r, StlKind::true_) || is(l, StlKind::true_)) return Stl(true);
        return l.release(r, interval());
    }
    case StlKind::true_:
    case StlKind::false_:
    case StlKind::comparison:
        return *this;
    }
    CORA_FAIL(Error::report_to_dev, "eliminate_true_false: unknown formula kind");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
