
#include <cmath>
#include <stdio.h>
#include <chrono>

int main(int argc, char** argv)
{
  // Beginning of interval 
  double a = 1.0;

  // End of interval
  double b = 2.0;

  // Number of sub-intervals.
  int N = 1;

 
// From Mathematica Exact value is Cos[1] - Cos[2] = 0.9564491424152821

  double exact =  0.9564491424152821;
 
 /********************
WARNING: As an experiment I am apparently pushing this example
 beyond the ability of double precesion. Why? Not sure!
*********************/
  
  for(int Nintervals; Nintervals < 16; Nintervals++)
    { N = 4*N;
    
  // Perform the integral
       // Pre-compute the window width.
  double h = (b-a)/((double)N);
   double sum = 0.0;

   // TIMING LINE 1: Get the starting timestamp. 
  std::chrono::time_point<std::chrono::steady_clock> begin_time =
    std::chrono::steady_clock::now();
   for (int i = 0; i < N; i++)
  {
    sum += h*sin(a + h*i + h*0.5);
  }
  // TIMING LINE 2: Get the ending timestamp.
  std::chrono::time_point<std::chrono::steady_clock> end_time =
    std::chrono::steady_clock::now();

  // TIMING LINE 3: Compute the difference.
  std::chrono::duration<double> difference_in_time = end_time - begin_time;
  //
  // TIMING LINE 4: Get the difference in seconds.
  double difference_in_seconds = difference_in_time.count();

  // Print the integral.
  printf("The integral from %.8f to %.8f of sin(x) using %d intervals is %.15f and error is %.10e  \n", a, b, N, sum, exact - sum);
  printf("This took %.8f seconds.\n", difference_in_seconds);
    }
  
  return 0;
}


