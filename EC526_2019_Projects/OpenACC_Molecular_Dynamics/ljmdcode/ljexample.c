/* 
 *  ljexample.c
 *
 *  Example of a Lennard-Jones simulation in C
 *  (problem 3 of assignment 3 in CHM1464)
 *
 *  Input (from standard input): 
 *
 *   number of particles 
 *   density of the system
 *   initial temperature (standard deviation of the velocities)
 *   runtime
 *   time step 
 *   random number generator seed
 *   equilibration time 
 *
 *  Output (from standard output): lines containing
 *
 *   time, energy H, pot. en. U, kin. en. K, temperature T, fluctuations
 *
 *  Notes:
 *
 *  - To compile: cc ljexample.c -o ljexample -lm -O3 -ffast-math
 *
 *  - To run: ljexample
 *        or: ljexample < input.ini > output.dat
 *    where input parameters are listed in the file "input.ini", 
 *    and output is redirected to the file "output.dat".
 *
 *  - Appropriate values for the equilibrium time are best found by
 *    doing a short run and seeing when the potential energy has reach a
 *    stationary value.
 *
 *  - All reported energies values are divided by the number of particles N.
 *
 *  - Fluctuations are the root mean square of H-<H>, with <H> the mean H.
 *  
 *  - This program is intended for small systems, so no attempt was
 *    made to implement cell divisions or Verlet neighbor lists.
 *
 *  Ramses van Zon, 13 November 2008
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* properties of the system */
const double rc   = 3.0;  /* outer cutoff radius */
const double rcp  = 2.5;  /* inner cutoff, or start of smoothing */
int     N;           /* number of particles */
double  rho;         /* density of the system */
double  L;           /* length of the box */
double  dt;          /* duration of single time step */
double  halfDt;      /* half the time step */
double  runtime;     /* how long to run */
long    seed;        /* random seed */
double  burninTime;  /* when to start measuring energies etc. */
double  K;           /* kinetic energy */
double  U;           /* potential energy */
double  H;           /* total energy */
double  T;           /* kinetic temperature */

/* structure for the properties of one atoms */
struct Atom
{
   double  rx, ry, rz;  /* position */
   double  px, py, pz;  /* momentum */
   double  fx, fy, fz;  /* force */
};

/* function to set up cubic lattice */
double latticex, latticey, latticez;
void makeLatticePosition(double a)
{
   static int i = 0;
   static int j = 0;
   static int k = 0;
   latticex = i*a - 0.5*L;
   latticey = j*a - 0.5*L;
   latticez = k*a - 0.5*L;
   i = i + 1;
   if ( i*a > L - 1.0e-6 )   /* next point outside box? */
   {
      i = 0;                 /* then put back */
      j = j + 1;             /* and move to next y level */

      if ( j*a > L - 1.0e-6 )  /* outside box? */
      {
         j = 0;                /* then put back */
         k = k + 1;            /* and move to next z level */

         if ( k*a > L - 1.0e-6 )  /* outside box? */
         {
            i = 0;                /* then back to the start */
            j = 0;
            k = 0;
         }
      }
   }
}

/* function dealing with periodic boundary conditions: */
/* increases or decreases *u until -L/2 <= *u < L/2 */
double makePeriodic(double u)
{
   while ( u < -0.5*L ) 
   {
      u = u + L;
   }

   while ( u >= 0.5*L ) 
   {
      u = u - L;
   }

   return u;
} 

/* function to compute forces */
void computeForces(struct Atom atoms[])
{
   int     i, j;                      /* particle indices */
   double  dx, dy, dz;                /* distance vector */
   double  r, r2, r2i, r6i;           /* distance, r^2, r^{-2} and r^{-6} */
   double  fij;                       /* force multiplier */
   double  eij;                       /* potential energy between i and j */
   double  x, alpha, dalpha;          /* auxiliar variable in smoothing */

   /* initialize energy and forces to zero */
   U = 0;

   for ( i = 0; i < N; i = i + 1 ) 
   {
      atoms[i].fx = 0;
      atoms[i].fy = 0;
      atoms[i].fz = 0;
   }

   /* determine interaction for each pair of particles (i,j) */
   for ( i = 0; i < N-1; i = i + 1 )
   {
      for ( j = i+1; j < N; j = j + 1 ) 
      {
         /* determine distance in periodic system */
         dx = makePeriodic(atoms[i].rx - atoms[j].rx);
         dy = makePeriodic(atoms[i].ry - atoms[j].ry);
         dz = makePeriodic(atoms[i].rz - atoms[j].rz);
         r2 = dx*dx + dy*dy + dz*dz; 

         /* note : using square distance saves taking a sqrt */
         if ( r2 < rc*rc ) 
         {
            r2i = 1/r2;
            r6i = r2i*r2i*r2i;
            fij = 48*r2i*r6i*(r6i-0.5);
            eij = 4*r6i*(r6i-1);

            /* within smooth cutoff region? */
            if ( r2 > rcp*rcp ) 
            {
               /* replace phi by alpha * phi                         */

               /* based on a slight rewriting of the alpha factor to */
               /*  alpha = 1/2 - 1/4 x (x^2 - 3)                     */
               /* where                                              */
               /*  x = (2 r - rcp - rc)/(rcp - rc)                   */

               r      = sqrt(r2);
               x      = (2*r - rcp - rc)/(rcp - rc);
               alpha  = 0.5 - 0.25*x*(x*x - 3);
               dalpha = 1.5*(x*x - 1)/(r*(rcp-rc));
               fij    = alpha*fij + dalpha*eij;
               eij    = alpha*eij;
            }

            atoms[i].fx = atoms[i].fx + fij*dx;
            atoms[i].fy = atoms[i].fy + fij*dy;
            atoms[i].fz = atoms[i].fz + fij*dz;
            atoms[j].fx = atoms[j].fx - fij*dx;
            atoms[j].fy = atoms[j].fy - fij*dy;
            atoms[j].fz = atoms[j].fz - fij*dz;
            U = U + eij;
         }
      }
   }
}

/* function for gaussian random variables */
double gaussian()
{
   static int    have = 0;
   static double x2;
   double fac, y1, y2, x1;
 
   if ( have == 1 )  /* already one available ? */
   {
      have = 0;
      return x2;
   } 
   else 
   {
      /* generate a pair of random variables */
      y1  = drand48();
      y2  = drand48();
      fac = sqrt(-2*log(y1));
      have = 1;
      x1 = fac*sin(2*M_PI*y2); /* x1 and x2 are now gaussian */
      x2 = fac*cos(2*M_PI*y2); /* so store one */
      return x1;               /* and return the other. */
   }
}

/* function to initialize the system */
void initialize(struct Atom atoms[])
{
   double  scale, a;
   int     i;

   /* generate positions */
   a = L/(int)(cbrt(N)+0.99999999999); /* lattice distance */

   for ( i = 0; i < N; i = i + 1 ) 
   {
      makeLatticePosition(a);
      atoms[i].rx = latticex;
      atoms[i].ry = latticey;
      atoms[i].rz = latticez;
   }

   /* generate momenta */
   srand48(seed);  /* initialized the random number generator used in gaussian */
   scale = sqrt(T);
   K     = 0;

   for ( i = 0; i < N; i = i + 1 ) 
   {
      atoms[i].px = scale*gaussian();
      atoms[i].py = scale*gaussian();
      atoms[i].pz = scale*gaussian();
      K = K 
         + atoms[i].px*atoms[i].px 
         + atoms[i].py*atoms[i].py 
         + atoms[i].pz*atoms[i].pz;
   }

   /* compute instantaneous kinetic temperature and energy */
   T = K/(3*N);
   K = K/2;

   /* compute force and potential energy U */
   computeForces(atoms);

   /* compute total energy */
   H = U + K;

   /* report results */ 
   printf("# time     E         U          K         T   <[H-<H>]^2>\n");
   printf("%8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n", 0., H/N, U/N, K/N, T, 0.);
}

/* Verlet integration step */
void integrateStep(struct Atom atoms[])
{
   int i;

   /* half-force step */
   for ( i = 0; i < N; i = i + 1 ) 
   {
      atoms[i].px = atoms[i].px + 0.5*dt*atoms[i].fx;
      atoms[i].py = atoms[i].py + 0.5*dt*atoms[i].fy;
      atoms[i].pz = atoms[i].pz + 0.5*dt*atoms[i].fz;
   }

   /* full free motion step */
   for ( i = 0; i < N; i = i + 1 ) 
   {
      atoms[i].rx = atoms[i].rx + dt*atoms[i].px;
      atoms[i].ry = atoms[i].ry + dt*atoms[i].py;
      atoms[i].rz = atoms[i].rz + dt*atoms[i].pz;
   }

   /* positions were changed, so recompute the forces */
   computeForces(atoms);

   /* final force half-step */
   K = 0;

   for ( i = 0; i < N; i = i + 1 ) 
   {
      atoms[i].px = atoms[i].px + 0.5*dt*atoms[i].fx;
      atoms[i].py = atoms[i].py + 0.5*dt*atoms[i].fy;
      atoms[i].pz = atoms[i].pz + 0.5*dt*atoms[i].fz;
      K = K 
         + atoms[i].px*atoms[i].px 
         + atoms[i].py*atoms[i].py 
         + atoms[i].pz*atoms[i].pz;
   }

   /* finish computing T, K, H */
   T = K/(3*N);
   K = K/2;
   H = U + K;
}

/* integration and measurement */
void run()
{
   struct  Atom atoms[N]; 
   int     numSteps    = (int)(runtime/dt + 0.5);
   int     burninSteps = (int)(burninTime/dt + 0.5);   
   int     count;          /* counts time steps */
   int     numPoints = 0;  /* counts measurements */
   double  sumH      = 0;  /* total energy accumulated over steps */
   double  sumH2     = 0;  /* total energy squared accumulated */
   double  avgH, avgH2, fluctH;  /* average energy, square, fluctuations */

   /* draw initial conditions */
   initialize(atoms);

   /* perform burn-in steps */
   for ( count = 0; count < burninSteps; count = count + 1 ) 
   {
      integrateStep(atoms);
      
      /* no measurement of energy fluctuations during equilibration */
      /* as this would pollute the averages */

      /* write energies to screen to track equilibration */ 
      printf("%8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n", 
             count*dt, H/N, U/N, K/N, T, 0.0);
   }

   /* perform rest of time steps */
   for ( count = burninSteps; count < numSteps; count = count + 1 ) 
   {
      /* perform integration step */
      integrateStep(atoms);

      /* accumulate energy and its square */
      sumH  = sumH + H;
      sumH2 = sumH2 + H*H;
      numPoints = numPoints + 1;

      /* determine averages and fluctuations */
      avgH   = sumH/numPoints;
      avgH2  = sumH2/numPoints;
      fluctH = sqrt(avgH2 - avgH*avgH);

      /* report results */ 
      printf("%8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n", 
             count*dt, H/N, U/N, K/N, T, fluctH/N);
   }
}

/* main program */
int main()
{
   /* read parameters */
   printf("#Enter number of particles (N): ");  
   fflush(stdout);
   scanf("%d", &N);

   printf("#Enter density (rho): ");  fflush(stdout);
   scanf("%lf", &rho);

   printf("#Enter initial temperature (T): ");
   fflush(stdout);
   scanf("%lf", &T);

   printf("#Enter runtime: ");  
   fflush(stdout);
   scanf("%lf", &runtime);

   printf("#Enter time step (dt): "); 
   fflush(stdout);
   scanf("%lf", &dt);

   printf("#Enter random seed: "); 
   fflush(stdout);
   scanf("%ld", &seed);

   printf("#Enter burn-in time (burninTime): ");  
   fflush(stdout);
   scanf("%lf", &burninTime);

   /* determine size of cubic box */
   L = cbrt(N/rho);

   /* report parameters */
   printf("\n#N=%d L=%lf T=%lf runtime=%lf dt=%lf seed=%ld burninTime=%lf\n",
          N, L, T, runtime, dt, seed, burninTime);

   /* run the simulation */
   run(); 

   return 0;
}
