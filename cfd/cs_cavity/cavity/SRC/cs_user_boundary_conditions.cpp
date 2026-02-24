/*============================================================================
 * Boundary conditions for the lid-driven cavity.
 *
 * Direct face-geometry version: classify each boundary face by its centroid
 * position, bypassing the zone-lookup mechanism entirely.
 *
 *   y > 0.999            → moving lid       u = (1, 0, 0)
 *   z < 0.001 or z>0.009 → symmetry planes
 *   else                 → no-slip walls     u = (0, 0, 0)
 *
 * Setting icodcl = CS_BC_DIRICHLET for wall faces is critical for laminar
 * flow: without it CS defaults to a wall-law BC (icodcl=5) which gives a
 * zero RHS norm when no turbulent viscosity is available.
 *============================================================================*/

#include "cs_headers.h"

#include <assert.h>
#include <math.h>
#include <string.h>

BEGIN_C_DECLS

/*----------------------------------------------------------------------------*/

void
cs_user_boundary_conditions([[maybe_unused]] cs_domain_t *domain,
                            [[maybe_unused]] int          bc_type[])
{
  const cs_mesh_t            *m  = domain->mesh;
  const cs_mesh_quantities_t *mq = domain->mesh_quantities;
  const cs_lnum_t n_b_faces = m->n_b_faces;

  /* Boundary face centroid coordinates [n_b_faces][3] */
  const cs_real_3_t *b_face_cog =
    (const cs_real_3_t *)mq->b_face_cog;

  cs_real_t *vel_rcodcl1 = CS_F_(vel)->bc_coeffs->rcodcl1;
  int       *vel_icodcl  = CS_F_(vel)->bc_coeffs->icodcl;

  int n_lid = 0, n_sym = 0, n_noslip = 0;

  for (cs_lnum_t f = 0; f < n_b_faces; f++) {

    const double fy = b_face_cog[f][1];
    const double fz = b_face_cog[f][2];

    if (fy > 0.999) {

      /* ── Moving lid: top wall y = 1, u = (1, 0, 0) ── */
      bc_type[f]                   = CS_SMOOTHWALL;
      vel_icodcl[f]                = CS_BC_DIRICHLET;   /* exact Dirichlet */
      vel_rcodcl1[0*n_b_faces + f] = 1.0;              /* u_x = 1 */
      vel_rcodcl1[1*n_b_faces + f] = 0.0;              /* u_y = 0 */
      vel_rcodcl1[2*n_b_faces + f] = 0.0;              /* u_z = 0 */
      n_lid++;

    }
    else if (fz < 0.001 || fz > 0.009) {

      /* ── Symmetry planes: z = 0 and z = 0.01 ── */
      bc_type[f] = CS_SYMMETRY;
      n_sym++;

    }
    else {

      /* ── No-slip walls: left (x=0), right (x=1), bottom (y=0) ── */
      bc_type[f]                   = CS_SMOOTHWALL;
      vel_icodcl[f]                = CS_BC_DIRICHLET;   /* exact Dirichlet */
      vel_rcodcl1[0*n_b_faces + f] = 0.0;
      vel_rcodcl1[1*n_b_faces + f] = 0.0;
      vel_rcodcl1[2*n_b_faces + f] = 0.0;
      n_noslip++;

    }
  }

  /* Diagnostic: print face counts on first time-step */
  if (domain->time_step->nt_cur <= 1) {
    bft_printf("  [cavity BC] lid=%d  symmetry=%d  no_slip=%d  total=%d\n",
               n_lid, n_sym, n_noslip, (int)n_b_faces);
    bft_printf("  [cavity BC] first lid face vel_rcodcl1[0] = %.6f\n",
               vel_rcodcl1[0]);
  }
}

END_C_DECLS
