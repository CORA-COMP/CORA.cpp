// exact_time - a time that is a sum of doubles, kept exactly
//
// The adaptive algorithms add their step sizes to the time. CORALean keeps that time a rational,
// so the time left, t_final - t, is exact there and is compared exactly with a step size. ExactTime
// is the same sum in fixed point: an integer count of 2^-1074 (the least double) in 36 words of
// two's complement, which holds every double and every sum or difference of them below 2^1229.
// Only the last step reads a double off the time left, as CORALean's ratUp at nearest does
// (rat_up); the times of the reachable set are the exact times rounded to nearest (nearest).
//
// Syntax:   ExactTime t;   t = t + dt;   bool last = ExactTime(t_final) - t < ExactTime(dt);
//           double h = (ExactTime(t_final) - t).rat_up();   double time = t.nearest();
//           std::string exact = t.text();   (the oracle's m*2^e)
// See also: cont_dynamics/nonlinear_sys/private/priv_adaptive_step.h
//
// CORA.cpp only: CORA adds the step sizes in doubles; CORALean's time is a rational.
// Authors:       Claude Opus 5.5
// Written:       16-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A sum of doubles, exactly: an integer number of 2^-1074 in two's complement.
class ExactTime {
  public:
    // Construction --------------------------------------------------------------------------------

    ExactTime() = default;

    /// The double x (finite) exactly: its 53-bit mantissa placed at its exponent.
    explicit ExactTime(double x) {
        if (x == 0 || !std::isfinite(x)) return;
        int e = 0;
        const double f = std::frexp(std::abs(x), &e); // |x| = f 2^e, f in [0.5, 1)
        auto mantissa = static_cast<std::uint64_t>(std::ldexp(f, 53));
        int at = e - 53 + kFraction; // the bit of the mantissa's last digit
        if (at < 0) {
            mantissa >>= -at; // a subnormal: the dropped bits are zero
            at = 0;
        }
        words_[std::size_t(at / 64)] = mantissa << (at % 64);
        if (at % 64 != 0) words_[std::size_t(at / 64) + 1] = mantissa >> (64 - at % 64);
        if (x < 0) *this = negated();
    }

    // Arithmetic ----------------------------------------------------------------------------------

    ExactTime operator+(const ExactTime &b) const {
        ExactTime out;
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < kWords; ++i) {
            const std::uint64_t s = words_[i] + carry;
            const std::uint64_t c = s < carry ? 1 : 0;
            out.words_[i] = s + b.words_[i];
            carry = c + (out.words_[i] < s ? 1 : 0);
        }
        return out;
    }
    ExactTime operator+(double x) const { return *this + ExactTime(x); }
    ExactTime operator-(const ExactTime &b) const { return *this + b.negated(); }

    /// k times the time, exactly (doubling and adding, from the highest bit of k).
    ExactTime times(std::uint32_t k) const {
        ExactTime out;
        for (int bit = 31; bit >= 0; --bit) {
            out = out + out;
            if ((k >> bit) & 1U) out = out + *this;
        }
        return out;
    }

    bool operator<(const ExactTime &b) const { return (*this - b).negative(); }
    bool operator<=(const ExactTime &b) const { return !(b < *this); }

    // The doubles and the text --------------------------------------------------------------------

    /// The nearest double (ties to even).
    double nearest() const {
        if (negative()) return -negated().nearest();
        const int top = highest_bit();
        if (top < 0) return 0.0;
        // the 53 bits from the highest, then the bit after them and whether any below is set
        const int low = top >= 52 ? top - 52 : 0;
        std::uint64_t mantissa = 0;
        for (int i = top; i >= low; --i) mantissa = 2 * mantissa + bit(i);
        bool sticky = false;
        for (int i = low - 2; i >= 0 && !sticky; --i) sticky = bit(i) != 0;
        const bool half = low > 0 && bit(low - 1) != 0;
        if (half && (sticky || (mantissa & 1U) != 0)) ++mantissa;
        return std::ldexp(static_cast<double>(mantissa), low - kFraction);
    }

    /// CORALean's ratUp at nearest of the time as a rational in lowest terms (0 below 0): the
    /// numerator rounded as it is built from its highest bit (v = 2 v + bit), over the denominator.
    double rat_up() const {
        const int top = highest_bit();
        if (negative() || top < 0) return 0.0;
        int zeros = 0;
        while (zeros < kFraction && bit(zeros) == 0) ++zeros; // the factors 2 of the numerator
        double numerator = 0;
        for (int i = top; i >= zeros; --i) {
            numerator = numerator + numerator;
            if (bit(i) != 0) numerator = numerator + 1.0;
        }
        return numerator / std::ldexp(1.0, kFraction - zeros);
    }

    /// The exact value as the oracle's text m*2^e: the integer of 2^-1074 in decimal.
    std::string text() const {
        if (negative()) return "-" + negated().text();
        // the decimal digits, nine at a time, from the remainders of division by 10^9
        std::array<std::uint64_t, kWords> rest = words_;
        std::string digits;
        const auto zero = [&] {
            for (const std::uint64_t w : rest)
                if (w != 0) return false;
            return true;
        };
        do {
            std::uint64_t remainder = 0;
            for (std::size_t i = kWords; i-- > 0;) {
                // the word in two halves, so that remainder * 2^32 + half fits in 64 bits
                const std::uint64_t high = (remainder << 32) | (rest[i] >> 32);
                const std::uint64_t low = ((high % kBillion) << 32) | (rest[i] & 0xffffffffU);
                rest[i] = ((high / kBillion) << 32) | (low / kBillion);
                remainder = low % kBillion;
            }
            std::string group = std::to_string(remainder);
            if (!zero()) group.insert(0, 9 - group.size(), '0');
            digits.insert(0, group);
        } while (!zero());
        return digits + "*2^-" + std::to_string(kFraction);
    }

  private:
    static constexpr std::uint64_t kBillion = 1000000000U;
    static constexpr std::size_t kWords = 36;
    static constexpr int kFraction = 1074;
    std::array<std::uint64_t, kWords> words_{}; ///< the least significant first

    std::uint64_t bit(int i) const { return (words_[std::size_t(i / 64)] >> (i % 64)) & 1U; }
    bool negative() const { return (words_.back() >> 63) != 0; }

    /// The index of the highest set bit, -1 for 0.
    int highest_bit() const {
        for (int i = int(kWords) * 64 - 1; i >= 0; --i)
            if (bit(i) != 0) return i;
        return -1;
    }

    ExactTime negated() const {
        ExactTime inverted;
        for (std::size_t i = 0; i < kWords; ++i) inverted.words_[i] = ~words_[i];
        ExactTime one;
        one.words_[0] = 1;
        return inverted + one;
    }
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
