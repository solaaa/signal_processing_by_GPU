
import reikna.cluda as cluda
import numpy as np
import pyopencl.array as clarray
import reikna.transformations as transformations
from pyopencl import clmath
import pyopencl as cl
import time as t


import numpy as np
import pyopencl as cl

def stat(thr):
    fftN = 10 
    width = 5
    prg = thr.compile("""
    //#include <math.h>
    KERNEL void summ(
        GLOBAL_MEM int *s_g,
        GLOBAL_MEM double *Xp_g)
    {

      int x = get_global_id(0);
      int y = get_global_id(1);
      int width = 1024;

      int Xp_i = (int)Xp_g[y*width + x];
      int Xp_i1 = (int)Xp_g[y*width + x + 1];

      if (Xp_i*width + x < 1000*width)
      {

          s_g[Xp_i*width + x] = s_g[Xp_i*width + x] + 1;
          //__syncthreads();
      }
      if(abs(Xp_i1 - Xp_i) > 10 && x < width-1)
		  {
		  	if(Xp_i1 - Xp_i > 0)
		  	{
		  		for(int j = 1; j < Xp_i1 - Xp_i; j++)
		  		{

		  			s_g[(Xp_i + j)*width + x] +=  1;
                                        //__syncthreads();
                                }
		  	}
		  	else
		  	{
		  		//a = -a;
		  		for(int j = 1; j <  Xp_i - Xp_i1; j++)
		  		{

		  			s_g[(Xp_i - j)*width + x] +=  1;
                                        //__syncthreads();
		  		}
		  	}
		  }


    }
    """)

    summ = prg.summ

    return summ

def logg10(thr):

    prg = thr.compile("""
    KERNEL void logg10(
        GLOBAL_MEM double *Xp_g)
    {
      int x = get_global_id(0);
      int y = get_global_id(1);
      int width = 1024;

      float LOG10E = 0.434294;
      int i = 0;
      float Xp = Xp_g[y*width + x];

      while (Xp > 1)
      {
        Xp = Xp/10.0;
        i = i + 1;
      }
      float logXp = 0;
      //float j = 1.0;
      float p1 = 1.0;
      float p2 = Xp - 1;
      float p3 = 1.0;
      float order;
      while (1)
      {
        // order = [(-1)^(n-1)]*(x^n)/n
        order = p1*p2/p3; 
        //order = pow((double)-1.0, (double)(j+1.0))*pow((double)(Xp - 1.0), (double)j)/(double)j;

        if(order>0 && order<0.0001 || order<0 && order>-0.0001)
        {
          break;
        }
        logXp = logXp + order;
        //j = j + 1.0;
        p1 = -p1;
        p2 = p2 * (Xp - 1);
        p3 = p3 + 1;
      }

      logXp = 10*(logXp*LOG10E + 3 + i);
      logXp = logXp * 10;
      Xp_g[y*width + x] = floor(logXp);
    }
    """)

    logg10 = prg.logg10

    return logg10
