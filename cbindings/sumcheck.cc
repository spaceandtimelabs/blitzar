#include "cbindings/fixed_pedersen.h"

#include "cbindings/backend.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// sxt_prove_sumcheck
//--------------------------------------------------------------------------------------------------
void sxt_prove_sumcheck(void* polynomials, void* evaluation_point, unsigned field_id,
                        const sumcheck_descriptor* descriptor, void* transcript_callback,
                        void* transcript_context) {
  auto backend = cbn::get_backend();
  backend->prove_sumcheck(polynomials, evaluation_point, field_id,
                          *reinterpret_cast<const cbnb::sumcheck_descriptor*>(descriptor),
                          transcript_callback, transcript_context);
}
