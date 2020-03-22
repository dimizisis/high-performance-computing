#include <graphics.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
 
#define pi M_PI

#define ITERATIONS 9000000
#define TRIANGLE_LEN 400
#define VERTICES_NUM 3

void draw_vertices(double vertices[VERTICES_NUM][VERTICES_NUM], double window_side);
void draw_triangle(double vertices[VERTICES_NUM][VERTICES_NUM]);
 
int main(void){
 
	double vertices[VERTICES_NUM][VERTICES_NUM], window_side=10+2*TRIANGLE_LEN, start, end;
 
	initwindow(window_side, window_side, "Chaos");

	draw_vertices(vertices, window_side);	// draw the vertices

	start = omp_get_wtime();

	draw_triangle(vertices);	// draw the triangle

	end = omp_get_wtime();
	
	(void) printf("\nTime elapsed: %g\n", (end-start));
 
	getch();	// do not close after completion
 
	closegraph();
 
	return 0;
}

/*
 * Function:  draw_vertices 
 * --------------------
 * Draws the vertices of triangle
 *
 *  vertices: the 2D array that will be filled with vertices
 *  window_side: width & height of graphics window
 *
 */

void draw_vertices(double vertices[VERTICES_NUM][VERTICES_NUM], double window_side){
	int i;
	for(i=0;i<VERTICES_NUM;++i){
		vertices[i][0] = window_side/2 + TRIANGLE_LEN*cos(i*2*pi/3);
		vertices[i][1] = window_side/2 + TRIANGLE_LEN*sin(i*2*pi/3);
		putpixel(vertices[i][0],vertices[i][1],15);
	}
}

/*
 * Function:  draw_triangle 
 * --------------------
 * Draws the triangle with given iterations & triangle length
 *
 *  vertices: the 2D array that will be filled with vertices
 *
 */

void draw_triangle(double vertices[VERTICES_NUM][VERTICES_NUM]){

	double seedX = rand()%(int)(vertices[0][0]/2 + (vertices[1][0] + vertices[2][0])/4);
	double seedY = rand()%(int)(vertices[0][1]/2 + (vertices[1][1] + vertices[2][1])/4);

	putpixel(seedX, seedY, 15);

	int i, choice;
	#pragma omp parallel for default(shared) private(i, choice, seedX, seedY) 
	for(i=0;i<ITERATIONS;++i){
		choice = rand()%3;
		seedX = (seedX + vertices[choice][0])/2;
		seedY = (seedY + vertices[choice][1])/2;
		putpixel(seedX, seedY, 15);
	}
}