#ifndef SYMNMF_H
#define SYMNMF_H

/* Memory and Utility Functions */
void error_and_exit(void);
double** allocate_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
void print_matrix(double** matrix, int rows, int cols);

/* Math Algorithms (Section 1 of PDF) */
double sq_euclidean_dist(double* p1, double* p2, int d);
double** sym(double** points, int n, int d);

#endif