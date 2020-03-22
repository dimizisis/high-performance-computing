#include <stdio.h> 
#include <graphics.h> 
#include <omp.h>

void DDA(int x0, int y0, int x1, int y1, int color);
int abs (int n);
 
int main(void) 
{ 
    int gd = DETECT, gm; 
  
    // Initialize graphics function 
    initgraph (&gd, &gm, NULL);    
  
    int x0 = 2, y0 = 2, x1 = 111, y1 = 200; 

    #pragma omp parallel sections
    {
        #pragma omp section
        DDA(x0, y0, (x0+x1)/2, (y0+y1)/2, RED); // draw until the midpoint of line

        #pragma omp section
        DDA(x1, y1, (x0+x1)/2, (y0+y1)/2, GREEN);   // draw from the midpoint till the end of line
    }
    getch();    // do not close program
    return 0; 
}  

/*
 * Function:  abs 
 * --------------------
 * Returns the absolute value of an integer
 *
 *  n: the integer whose absolute value will be returned
 *
 */

int abs (int n) { 
    return ((n>0) ? n : (n * (-1))); 
} 
  
/*
 * Function:  DDA 
 * --------------------
 * Given 2 points, performs DDA algorithm, in order to draw line
 *
 *  x0: starting point (x axis)
 *  y0: starting point (y axis)
 *  x1: end point (x axis)
 *  y1: end point (y axis)
 *  color: color of line (BGI)
 *
 */

void DDA(int x0, int y0, int x1, int y1, int color){ 

    int i;
    // calculate dx & dy 
    int dx = x1 - x0; 
    int dy = y1 - y0; 
  
    // calculate steps required for generating pixels 
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy); 
  
    // calculate increment in x & y for each steps 
    float x_inc = dx / (float) steps; 
    float y_inc = dy / (float) steps; 

    // Put pixel for each step 
    float x = x0; 
    float y = y0; 

    for(i=0; i<=steps; ++i){ 
        putpixel (x,y,color);  // put pixel at (x,y) 
        x += x_inc;           // increment in x at each step 
        y += y_inc;           // increment in y at each step
        // delay(100);
    } 

}