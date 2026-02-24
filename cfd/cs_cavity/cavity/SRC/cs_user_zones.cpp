/*============================================================================
 * Boundary zone definitions for the lid-driven cavity.
 *
 * Geometric criteria on face CENTRES (dx = dy = 0.01):
 *
 *   "top_lid"  : y > 0.999        → y = 1.0 plane only   (100 faces)
 *   "no_slip"  : y < 0.001 or x < 0.001 or x > 0.999
 *                                  → bottom + left + right (300 faces)
 *   "symmetry" : z < 0.001 or z > 0.009
 *                                  → front z=0 and back z=0.01 (20000 faces)
 *
 * Tolerances chosen so each zone is exclusive:
 *   top_lid    faces have y_center = 1.000 → 1.000 > 0.999 (pass)
 *              vs left/right wall top faces y_center = 0.995 < 0.999 (fail)
 *   no_slip    left faces have x_center = 0.000, right = 1.000
 *              vs top_lid faces x_center ∈ [0.005, 0.995] (fail)
 *============================================================================*/

#include "cs_headers.h"

BEGIN_C_DECLS

void
cs_user_zones(void)
{
  cs_boundary_zone_define("top_lid",
                          "y > 0.999",
                          0);

  cs_boundary_zone_define("no_slip",
                          "y < 0.001 or x < 0.001 or x > 0.999",
                          0);

  cs_boundary_zone_define("symmetry",
                          "z < 0.001 or z > 0.009",
                          0);
}

END_C_DECLS
