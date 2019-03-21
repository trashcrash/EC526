#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <complex>
using namespace std;
#include "fft.h"

int main()
{
  int N = 16;
  Complex * omega = new Complex[N];
  Complex * F = new Complex[N];
  Complex * Fnew = new Complex[N];
  Complex * Ftilde  = new Complex[N];
  
  makePhase(omega,N);

  /* Test slow FT */
  for(int x = 0; x < N; x++){
    F[x] = 2* sin( 2.0*PI*x/(double) N) + 4*cos( 2.0*PI*3.0*x/(double) N);
    //    cout<<" x = "<< x << "  F =    " <<  F[x]  << endl;
  }
  
  FT(Ftilde, F,omega,N);

  for(int k = 0; k < N; k++)
    cout<<" k = "<< k << "  Ftilde =   " <<  Ftilde[k]  << endl;

  FTinv(Fnew,Ftilde, omega, N);

  for(int x = 0; x < N; x++) 
    cout<<" x = "<< x << "  F =    " <<  F[x]  << " : " << Fnew[x] <<  endl;

  /*  Place to write and test a recursive FFT */
  //FFT(*Ftiled, F, N);
  //FFTinv(*Ftilde,F, N);

  return 0;
}
