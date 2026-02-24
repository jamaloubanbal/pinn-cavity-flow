/*============================================================================
 * User parameters for the lid-driven cavity (Re = 100).
 *
 * Fluid:  rho = 1 kg/m³,  mu = 0.01 Pa·s  →  Re = rho*U*L/mu = 100
 * Domain: unit square [0,1]²,  U_lid = 1 m/s
 * Time:   dt = 0.005 s,  t_end = 25 s  (5000 steps)
 *
 * Turbulence model is set to "off" (laminar) via setup.xml.
 * Fluid properties are also set in setup.xml; re-confirmed here for safety.
 *============================================================================*/

#include "cs_headers.h"
#include <math.h>

BEGIN_C_DECLS

/*----------------------------------------------------------------------------*/
/* Select physical model options — called before any field allocation.        */
/*----------------------------------------------------------------------------*/

void
cs_user_model(void)
{
  /* Force laminar — belt and suspenders, in case setup.xml is ignored. */
  cs_turb_model_t *tm = cs_get_glob_turb_model();
  tm->model = CS_TURB_NONE;
}

/*----------------------------------------------------------------------------*/
/* General numerical and physical parameters.                                  */
/*----------------------------------------------------------------------------*/

void
cs_user_parameters([[maybe_unused]] cs_domain_t *domain)
{
  /* ── time stepping ──────────────────────────────────────────────────────── */
  const cs_real_t dt_ref = 0.005;
  domain->time_step->dt_ref = dt_ref;
  domain->time_step->nt_max = (int)(25.0 / dt_ref);   /* 5000 steps → t = 25 s */

  /* ── fluid properties — rho = 1, mu = 0.01  →  Re = 100 ───────────────── */
  cs_fluid_properties_t *fp = cs_get_glob_fluid_properties();
  fp->ro0    = 1.0;
  fp->viscl0 = 0.01;
  fp->p0     = 0.0;
}

END_C_DECLS
