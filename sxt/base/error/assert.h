/**
 * Adopted from Envoy
 *
 * See third_party/license/envoy.LICENSE
 */
#pragma once

#include <cstdlib>
#include <iostream>

//--------------------------------------------------------------------------------------------------
// Helper Macros
//--------------------------------------------------------------------------------------------------
#define _ASSERT_EXPAND(X) X
#define _ASSERT_SELECTOR(_1, _2, ASSERT_MACRO, ...) ASSERT_MACRO

//--------------------------------------------------------------------------------------------------
// _NULL_ASSERT_IMPL
//--------------------------------------------------------------------------------------------------
// This non-implementation ensures that its argument is a valid expression that can be statically
// casted to a bool, but the expression is never evaluated and will be compiled away.
#define _NULL_ASSERT_IMPL(X, ...)                                                                  \
  do {                                                                                             \
    constexpr bool __assert_dummy_variable = false && static_cast<bool>(X);                        \
    (void)__assert_dummy_variable;                                                                 \
  } while (false);

//--------------------------------------------------------------------------------------------------
// _ASSERT_IMPL
//--------------------------------------------------------------------------------------------------
// CONDITION_STR is needed to prevent macros in condition from being expected, which obfuscates
// the logged failure, e.g., "EAGAIN" vs "11".
#define _ASSERT_IMPL(CONDITION, CONDITION_STR, MESSAGE)                                            \
  do {                                                                                             \
    if (!(CONDITION)) {                                                                            \
      std::cerr << __builtin_FILE() << ":" << __builtin_LINE() << " failed assert: [ "             \
                << CONDITION_STR << " ]. " << MESSAGE << "\n";                                     \
      std::abort();                                                                                \
    }                                                                                              \
  } while (false);

//--------------------------------------------------------------------------------------------------
// SXT_RELEASE_ASSERT
//--------------------------------------------------------------------------------------------------
#define _RELEASE_ASSERT_ORIGINAL(X) _ASSERT_IMPL(X, #X, "")
#define _RELEASE_ASSERT_VERBOSE(X, Y) _ASSERT_IMPL(X, #X, Y)

// If SXT_RELEASE_ASSERT is called with one argument, the _ASSERT_SELECTOR will return
// _RELEASE_ASSERT_ORIGINAL and this will call _RELEASE_ASSERT_ORIGINAL(__VA_ARGS__).
//
// If SXT_RELEASE_ASSERT is called with two arguments, _ASSERT_SELECTOR will return
// _RELEASE_ASSERT_VERBOSE, and this will call _RELEASE_ASSERT_VERBOSE,(__VA_ARGS__)
#define SXT_RELEASE_ASSERT(...)                                                                    \
  _ASSERT_EXPAND(_ASSERT_SELECTOR(__VA_ARGS__, _RELEASE_ASSERT_VERBOSE,                            \
                                  _RELEASE_ASSERT_ORIGINAL)(__VA_ARGS__))

//--------------------------------------------------------------------------------------------------
// SXT_DEBUG_ASSERT
//--------------------------------------------------------------------------------------------------
#ifdef NDEBUG
// Use empty assert in release mode
#define _DEBUG_ASSERT_ORIGINAL(CONDITION) _NULL_ASSERT_IMPL(CONDITION)
#define _DEBUG_ASSERT_VERBOSE(CONDITION, MESSAGE) _NULL_ASSERT_IMPL(CONDITION, MESSAGE)
#else
// Use assert implementation in debug mode
#define _DEBUG_ASSERT_ORIGINAL(CONDITION) _ASSERT_IMPL(CONDITION, #CONDITION, "")
#define _DEBUG_ASSERT_VERBOSE(CONDITION, MESSAGE) _ASSERT_IMPL(CONDITION, #CONDITION, MESSAGE)
#endif

// If SXT_DEBUG_ASSERT is called with one argument, the _ASSERT_SELECTOR will return
// _DEBUG_ASSERT_ORIGINAL and this will call _DEBUG_ASSERT_ORIGINAL(__VA_ARGS__).
//
// If SXT_DEBUG_ASSERT is called with two arguments, _ASSERT_SELECTOR will return
// _DEBUG_ASSERT_VERBOSE, and this will call _DEBUG_ASSERT_VERBOSE,(__VA_ARGS__)
#define SXT_DEBUG_ASSERT(...)                                                                      \
  _ASSERT_EXPAND(                                                                                  \
      _ASSERT_SELECTOR(__VA_ARGS__, _DEBUG_ASSERT_VERBOSE, _DEBUG_ASSERT_ORIGINAL)(__VA_ARGS__))
