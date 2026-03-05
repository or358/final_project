#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

/* --- Utility & Memory Functions --- */

void error_and_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

double** allocate_matrix(int rows, int cols) {
    double** matrix;
    int i;
    matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) error_and_exit();
    
    for (i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) error_and_exit();
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    int i;
    if (matrix == NULL) return;
    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void print_matrix(double** matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) printf(",");
        }
        printf("\n");
    }
}

/* --- Section 1.1: The Similarity Matrix --- */

double sq_euclidean_dist(double* p1, double* p2, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

double** sym(double** points, int n, int d) {
    double** A = allocate_matrix(n, n);
    int i, j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                A[i][j] = exp(-sq_euclidean_dist(points[i], points[j], d) / 2.0);
            }
        }
    }
    return A;
}