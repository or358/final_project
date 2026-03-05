#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> /* <-- Add this line for strcmp */
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

/* --- Section 1.2: The Diagonal Degree Matrix --- */
double** ddg(double** A, int n) {
    double** D = allocate_matrix(n, n);
    int i, j;
    double sum;
    
    for (i = 0; i < n; i++) {
        sum = 0.0;
        /* Calculate the sum of row i in matrix A */
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        
        /* Fill row i in matrix D: only the diagonal gets the sum */
        for (j = 0; j < n; j++) {
            if (i == j) {
                D[i][j] = sum;
            } else {
                D[i][j] = 0.0;
            }
        }
    }
    return D;
}

/* --- Section 1.3: The Normalized Similarity Matrix --- */
double** norm(double** A, double** D, int n) {
    double** W = allocate_matrix(n, n);
    int i, j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            /* Prevent division by zero just in case */
            if (D[i][i] > 0 && D[j][j] > 0) {
                W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]);
            } else {
                W[i][j] = 0.0;
            }
        }
    }
    return W;
}

/* --- Matrix Math Helpers --- */
void mult_mat(double** A, double** B, double** C, int rowsA, int colsA, int colsB) {
    int i, j, k;
    for (i = 0; i < rowsA; i++) {
        for (j = 0; j < colsB; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void transpose(double** A, double** T, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            T[j][i] = A[i][j];
        }
    }
}

/* --- Section 1.4: Optimize H --- */
double** symnmf_optimize(double** W, double** H, int n, int k) {
    double diff = 1.0;
    int iter = 0, i, j;
    double **WH, **Ht, **HtH, **HHtH, **H_new;

    /* Allocate memory once to avoid leaks in the loop */
    WH = allocate_matrix(n, k);
    Ht = allocate_matrix(k, n);
    HtH = allocate_matrix(k, k);
    HHtH = allocate_matrix(n, k);
    H_new = allocate_matrix(n, k);

    while (iter < MAX_ITER && diff >= EPSILON) {
        mult_mat(W, H, WH, n, n, k);
        transpose(H, Ht, n, k);
        mult_mat(Ht, H, HtH, k, n, k);
        mult_mat(H, HtH, HHtH, n, k, k); /* Faster: H * (H^T * H) */

        diff = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_new[i][j] = H[i][j] * (1.0 - BETA + BETA * (WH[i][j] / HHtH[i][j]));
                diff += (H_new[i][j] - H[i][j]) * (H_new[i][j] - H[i][j]);
            }
        }
        
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H[i][j] = H_new[i][j];
            }
        }
        iter++;
    }

    /* Free intermediate memory */
    free_matrix(WH, n); free_matrix(Ht, k); free_matrix(HtH, k);
    free_matrix(HHtH, n); free_matrix(H_new, n);
    
    return H;
}

/* --- Section 2.2: C Interface (CLI) --- */

/* Helper to count points and dimensions from file */
void get_dimensions(char* filename, int* n, int* d) {
    FILE *fp = fopen(filename, "r");
    char line[2048];
    int cols = 0, rows = 0, i;
    if (!fp) error_and_exit();
    
    if (fgets(line, 2048, fp) != NULL) {
        rows++;
        cols = 1;
        for (i = 0; line[i] != '\0'; i++) {
            if (line[i] == ',') cols++;
        }
    }
    while (fgets(line, 2048, fp) != NULL) {
        rows++;
    }
    fclose(fp);
    *n = rows;
    *d = cols;
}

/* Helper to load the data into a C matrix */
double** load_data(char* filename, int n, int d) {
    FILE *fp = fopen(filename, "r");
    double **points = allocate_matrix(n, d);
    int i, j;
    if (!fp) error_and_exit();
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            if (fscanf(fp, "%lf", &points[i][j]) != 1) error_and_exit();
            if (j < d - 1) fgetc(fp); /* consume the comma */
        }
    }
    fclose(fp);
    return points;
}

int main(int argc, char **argv) {
    int n, d;
    char *goal, *filename;
    double **points, **A, **D, **W;
    
    if (argc != 3) error_and_exit();
    goal = argv[1];
    filename = argv[2];
    
    get_dimensions(filename, &n, &d);
    points = load_data(filename, n, d);
    
    if (strcmp(goal, "sym") == 0) {
        A = sym(points, n, d);
        print_matrix(A, n, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        A = sym(points, n, d);
        D = ddg(A, n);
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        A = sym(points, n, d);
        D = ddg(A, n);
        W = norm(A, D, n);
        print_matrix(W, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free_matrix(W, n);
    } else {
        error_and_exit();
    }
    
    free_matrix(points, n);
    return 0;
}