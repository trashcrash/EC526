#include <cmath>
#include <stdio.h>
#include <chrono>
#include <quadmath.h>



int main(int argc, char** argv)
{
 FILE *outfile;
 outfile = fopen("scaling.dat","w");

  // Beginning of interval 
  double a = 1.0;

  // End of interval
  double b = 2.0;

  // Number of sub-intervals.
  long int N = 1;

 
// From Mathematica Exact value is Cos[1] - Cos[2] = 0.9564491424152821

  double  exact =  0.9564491424152821043985048369437387934983;
  __float128 exact_quad = 0.9564491424152821043985048369437387934983Q;


 /********************
WARNING: As an experiment I am apparently pushing this example
 beyond the ability of double precesion. Why? Not sure!
*********************/
  
  for(long int Nintervals = 0; Nintervals < 15; Nintervals++)
    { N = 4*N;
    
  // Perform the integral
       // Pre-compute the window width.
  double h = (b-a)/((double)N);
  double sum = 0.0;
  __float128 sum_quad = 0.0;

   // TIMING LINE 1: Get the starting timestamp. 
  std::chrono::time_point<std::chrono::steady_clock> begin_time =
  std::chrono::steady_clock::now();
  
   for (long int i = 0; i < N; i++)
  {
    sum += h*sin(a + h*i + h*0.5);
    //   sum_quad +=   h_quad*sin(a_quad + h_quad*i + h_quad*0.5);
  }
  // TIMING LINE 2: Get the ending timestamp.
  std::chrono::time_point<std::chrono::steady_clock> end_time =
    std::chrono::steady_clock::now();

  // TIMING LINE 3: Compute the difference.
  std::chrono::duration<double> difference_in_time = end_time - begin_time;
  //
  // TIMING LINE 4: Get the difference in seconds.
  double difference_in_seconds = difference_in_time.count();

   // TIMING LINE 1: Get the starting timestamp. 
   begin_time = std::chrono::steady_clock::now();
  
   for (long int i = 0; i < N; i++)
  {
    //  sum += h*sin(a + h*i + h*0.5);
    sum_quad +=   h*sin(a  +  h * i + h*0.5);
  }

   // TIMING LINE 2: Get the ending timestamp.
    end_time = std::chrono::steady_clock::now();

  // TIMING LINE 3: Compute the difference.
   difference_in_time = end_time - begin_time;

   //
  // TIMING LINE 4: Get the difference in seconds.
   double difference_in_seconds_quad = difference_in_time.count();
  
  // Print the integral.
  printf("The integral from %38f to %.8f of sin(x) usng %ld intervals is %.15f and error is %.10e ", a, b, N, sum,  std::abs(exact - sum));
  printf("Quad error is %.30e \n", (double)(sum_quad - exact_quad));


  /***********************

Simple example of printing columns of data to file that is freiendly to
later gnuplot or Mathematica Plotting!  Of course you  will want to
restructure this according to what you want to plot but because
both plotting can easily select columns and rages it is a very flexible.

***********************/
  printf("This took %.8f seconds for double and  %.8f seconds for double quad \n", difference_in_seconds,difference_in_seconds_quad);
   
 fprintf(outfile," %ld  %.15f  %.10e  %.10e \n", N, sum, std::abs(exact -sum),(double)( sum_quad - exact_quad ));
   
 }

  fclose(outfile);

  return 0;
}


