#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "symnmf.h"

/* * פונקציות לניהול זיכרון - הקצאה ושחרור של מטריצות דו-ממדיות
 */

double** allocate_matrix(int rows, int cols) {
    double** matrix;
    int i;
    
    matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    
    for (i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            printf("An Error Has Occurred\n");
            exit(1);
        }
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

/* * פונקציית עזר להדפסת המטריצה בפורמט הנדרש (4 ספרות אחרי הנקודה)
 */
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

/* * האלגוריתם הראשון: חישוב מטריצת הדמיון (Similarity Matrix)
 */

double squared_euclidean_distance(double* p1, double* p2, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sum;
}

double** calculate_sym(double** points, int n, int d) {
    double** A;
    int i, j;
    double dist_sq;
    
    A = allocate_matrix(n, n);
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                dist_sq = squared_euclidean_distance(points[i], points[j], d);
                A[i][j] = exp(-dist_sq / 2.0);
            }
        }
    }
    return A;
}
