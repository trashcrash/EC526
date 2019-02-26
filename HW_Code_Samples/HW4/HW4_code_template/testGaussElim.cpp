#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include  "gaussElim.h"


int main()
{
  int i;
  double** A;
  double* b;
  int N = 3;
  
  A = new double*[N];
  
  for (i=0;i<N;i++) {
    A[i] = new double[N]; } b = new double[N];
  
  A[0][0] = -1; A[0][1] = 2; A[0][2] = -5; b[0] = 17;
  A[1][0] = 2; A[1][1] = 1; A[1][2] = 3; b[1] = 0;
  A[2][0] = 4; A[2][1] = -3; A[2][2] = 1; b[2] = -10;
  
  gaussianElimination(A,b,N);
  
for (i=0;i<N;i++)
  {
    printf("%f     \n", b[i]);
  }
 delete[] b;
 
for (i = 0; i < N; i++)
  {
    delete[] A[i];
  }
 
 delete[] A;
 
 return 0;
}
