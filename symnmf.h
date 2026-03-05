#ifndef SYMNMF_H
#define SYMNMF_H

/* Algorithm Constants */
#define BETA 0.5
#define EPSILON 1e-4
#define MAX_ITER 300

/* Memory and Utility Functions */
void error_and_exit(void);
double** allocate_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
void print_matrix(double** matrix, int rows, int cols);

/* Math Algorithms (Section 1 of PDF) */
double sq_euclidean_dist(double* p1, double* p2, int d);
double** sym(double** points, int n, int d);
double** ddg(double** A, int n);
double** norm(double** A, double** D, int n);

/* Algorithm 1.4: Optimize H */
void mult_mat(double** A, double** B, double** C, int rowsA, int colsA, int colsB);
void transpose(double** A, double** T, int rows, int cols);
double** symnmf_optimize(double** W, double** H, int n, int k);

#endif