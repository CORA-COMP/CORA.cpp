// set - cora::Set, a set whose class is known only at run time
//
// The result of an operation whose class depends on its input (a conversion, a projection, a
// binary operation between two classes) is a Set: an immutable value that holds any set class
// and offers what ContSet offers. is<T>() asks for the class, as<T>() gives it.
//
// Syntax:   Set S = Z.project({0, 1});   if (S.is<Zonotope>()) S.as<Zonotope>().generators();
//           Set T = S + Set(I);   Interval B = S->interval();
// See also: cont_set/cont_set.h, cont_set/registry.h
//
// CORA.cpp only: MATLAB is dynamically typed and needs no wrapper for a set of unknown class.
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Set {
  public:
    /// Holds a copy of any set (moved in where possible).
    template <class S, class = std::enable_if_t<std::is_base_of_v<ContSet, std::decay_t<S>> &&
                                                !std::is_same_v<std::decay_t<S>, Set>>>
    Set(S &&set) : set_(std::make_shared<const std::decay_t<S>>(std::forward<S>(set))) {}

    /// Shares a set that is owned elsewhere (a Python object keeps itself alive through `set`).
    explicit Set(std::shared_ptr<const ContSet> set) : set_(std::move(set)) {}

    const ContSet &get() const noexcept { return *set_; }
    operator const ContSet &() const noexcept { return *set_; }
    const ContSet *operator->() const noexcept { return set_.get(); }
    const ContSet &operator*() const noexcept { return *set_; }

    ClassId class_id() const noexcept { return set_->class_id(); }
    std::string class_name() const { return set_->class_name(); }

    /// Whether the set is of class S.
    template <class S>
    bool is() const noexcept {
        return set_->class_id() == S::kClassId;
    }

    /// The set as its class S; an error if it is of another class.
    template <class S>
    const S &as() const {
        if (!is<S>())
            CORA_FAIL(Error::wrong_value, "the set is a " + class_name() + ", not a " +
                                              cora::class_name(S::kClassId) +
                                              "; ask with is<...>() first");
        return static_cast<const S &>(*set_);
    }

  private:
    std::shared_ptr<const ContSet> set_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
