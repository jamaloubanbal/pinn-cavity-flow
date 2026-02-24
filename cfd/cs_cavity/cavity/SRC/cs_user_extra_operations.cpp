/*============================================================================
 * Extract centreline profiles for the lid-driven cavity comparison.
 *
 * At the end of the calculation, writes two CSV files:
 *   vertical_centreline.csv   — cells near x = 0.5, columns: y, u, v
 *   horizontal_centreline.csv — cells near y = 0.5, columns: x, u, v
 *
 * The files are written to the run directory (RESU/<run>/postprocessing/).
 *============================================================================*/

#include "cs_headers.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

BEGIN_C_DECLS

/*----------------------------------------------------------------------------*/

void
cs_user_extra_operations_initialize([[maybe_unused]] cs_domain_t *domain)
{
}

void
cs_user_extra_operations([[maybe_unused]] cs_domain_t *domain)
{
}

/*----------------------------------------------------------------------------*/
/*
 * Write centreline CSVs at the very end of the run.
 */
/*----------------------------------------------------------------------------*/

void
cs_user_extra_operations_finalize([[maybe_unused]] cs_domain_t *domain)
{
  const cs_mesh_t           *m  = domain->mesh;
  const cs_mesh_quantities_t *mq = domain->mesh_quantities;

  const cs_lnum_t n_cells = m->n_cells;
  const cs_real_3_t *cell_cen = (const cs_real_3_t *)mq->cell_cen;

  /* Velocity field: val[3*cell_id + component] */
  const cs_real_t *vel = CS_F_(vel)->val;

  /* tolerance: half a cell width for a 100×100 mesh (dx = 0.01) */
  const double tol = 0.006;

  /* ── Collect and sort cells for each centreline ──────────────────────── */

  typedef struct { double coord; double u; double v; } LinePoint;

  LinePoint *vc_pts = NULL;
  LinePoint *hc_pts = NULL;
  int n_vc = 0, n_hc = 0;
  int cap_vc = 256, cap_hc = 256;

  vc_pts = (LinePoint *)malloc(cap_vc * sizeof(LinePoint));
  hc_pts = (LinePoint *)malloc(cap_hc * sizeof(LinePoint));

  for (cs_lnum_t c = 0; c < n_cells; c++) {
    double x = cell_cen[c][0];
    double y = cell_cen[c][1];
    double u = vel[3 * c + 0];
    double v = vel[3 * c + 1];

    /* vertical centreline: x ≈ 0.5 */
    if (fabs(x - 0.5) < tol) {
      if (n_vc >= cap_vc) {
        cap_vc *= 2;
        vc_pts = (LinePoint *)realloc(vc_pts, cap_vc * sizeof(LinePoint));
      }
      vc_pts[n_vc++] = (LinePoint){ y, u, v };
    }

    /* horizontal centreline: y ≈ 0.5 */
    if (fabs(y - 0.5) < tol) {
      if (n_hc >= cap_hc) {
        cap_hc *= 2;
        hc_pts = (LinePoint *)realloc(hc_pts, cap_hc * sizeof(LinePoint));
      }
      hc_pts[n_hc++] = (LinePoint){ x, u, v };
    }
  }

  /* Simple insertion sort by coord (small arrays ~100 points each) */
  for (int i = 1; i < n_vc; i++) {
    LinePoint tmp = vc_pts[i];
    int j = i - 1;
    while (j >= 0 && vc_pts[j].coord > tmp.coord) {
      vc_pts[j + 1] = vc_pts[j]; j--;
    }
    vc_pts[j + 1] = tmp;
  }
  for (int i = 1; i < n_hc; i++) {
    LinePoint tmp = hc_pts[i];
    int j = i - 1;
    while (j >= 0 && hc_pts[j].coord > tmp.coord) {
      hc_pts[j + 1] = hc_pts[j]; j--;
    }
    hc_pts[j + 1] = tmp;
  }

  /* ── Write vertical centreline (x ≈ 0.5): y, u, v ────────────────────── */
  {
    FILE *fp = fopen("vertical_centreline.csv", "w");
    if (fp != NULL) {
      fprintf(fp, "y,u,v\n");
      for (int i = 0; i < n_vc; i++)
        fprintf(fp, "%.8f,%.8f,%.8f\n",
                vc_pts[i].coord, vc_pts[i].u, vc_pts[i].v);
      fclose(fp);
    }
  }

  /* ── Write horizontal centreline (y ≈ 0.5): x, u, v ──────────────────── */
  {
    FILE *fp = fopen("horizontal_centreline.csv", "w");
    if (fp != NULL) {
      fprintf(fp, "x,u,v\n");
      for (int i = 0; i < n_hc; i++)
        fprintf(fp, "%.8f,%.8f,%.8f\n",
                hc_pts[i].coord, hc_pts[i].u, hc_pts[i].v);
      fclose(fp);
    }
  }

  free(vc_pts);
  free(hc_pts);
}

END_C_DECLS
