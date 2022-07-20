#pragma once

#ifdef SXT_USE_CALLGRIND
#include <valgrind/callgrind.h>

#define SXT_TOGGLE_COLLECT CALLGRIND_TOGGLE_COLLECT
#else
#define SXT_TOGGLE_COLLECT
#endif
