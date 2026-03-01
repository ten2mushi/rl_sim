/**
 * Shared Fast Float Parsing Utility
 *
 * Locale-independent float parsing for OBJ and MTL parsers.
 * Static inline to avoid link-time duplication issues.
 */

#ifndef PARSE_UTILS_H
#define PARSE_UTILS_H

#include <stdbool.h>

/**
 * Parse float without locale overhead.
 * Returns pointer to character after parsed number.
 */
static inline const char* parse_float_fast(const char* str, float* out) {
    const char* p = str;
    float sign = 1.0f;
    float value = 0.0f;
    float fraction = 0.0f;
    float divisor = 10.0f;
    bool has_digits = false;

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;

    /* Sign */
    if (*p == '-') {
        sign = -1.0f;
        p++;
    } else if (*p == '+') {
        p++;
    }

    /* Integer part */
    while (*p >= '0' && *p <= '9') {
        value = value * 10.0f + (float)(*p - '0');
        has_digits = true;
        p++;
    }

    /* Decimal part */
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') {
            fraction += (float)(*p - '0') / divisor;
            divisor *= 10.0f;
            has_digits = true;
            p++;
        }
    }

    /* Exponent part (scientific notation) */
    if ((*p == 'e' || *p == 'E') && has_digits) {
        p++;
        int exp_sign = 1;
        int exp_value = 0;

        if (*p == '-') {
            exp_sign = -1;
            p++;
        } else if (*p == '+') {
            p++;
        }

        while (*p >= '0' && *p <= '9') {
            exp_value = exp_value * 10 + (*p - '0');
            p++;
        }

        float multiplier = 1.0f;
        for (int i = 0; i < exp_value; i++) {
            if (exp_sign > 0) {
                multiplier *= 10.0f;
            } else {
                multiplier *= 0.1f;
            }
        }
        value = (value + fraction) * multiplier;
    } else {
        value = value + fraction;
    }

    *out = sign * value;
    return has_digits ? p : str;
}

#endif /* PARSE_UTILS_H */
