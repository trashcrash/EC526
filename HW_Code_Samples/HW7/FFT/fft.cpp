#include "fft.h"

void makePhase(Complex *omega, int N )
{
  for(int k = 0; k < N; k++)
    omega[k] = exp(2.0*PI*I*(double)k/(double) N);
}

void FT(Complex * Ftilde, Complex * F, Complex * omega, int N)
{
  for(int k = 0; k < N; k++)
    {
      Ftilde[k] = 0.0;
      for(int x = 0; x < N; x++)
	Ftilde[k] += pow(omega[k],x)*F[x];
    }
}

void FTinv(Complex * F, Complex * Ftilde, Complex * omega, int N)
{
  for(int x = 0; x < N; x++)
    {
      F[x] = 0.0;
      for(int k = 0; k < N; k++)
	F[x] +=pow(omega[k],-x)*Ftilde[k]/(double) N;
    }
}
