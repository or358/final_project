#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* --- Helper Functions: Convert Python List <--> C Matrix --- */

/* Convert Python list of lists to C double** matrix */
double** convert_to_c_matrix(PyObject* py_list, int rows, int cols) {
    double** c_matrix;
    int i, j;
    PyObject *py_row, *py_float;

    c_matrix = allocate_matrix(rows, cols);
    
    for (i = 0; i < rows; i++) {
        py_row = PyList_GetItem(py_list, i);
        for (j = 0; j < cols; j++) {
            py_float = PyList_GetItem(py_row, j);
            c_matrix[i][j] = PyFloat_AsDouble(py_float);
        }
    }
    return c_matrix;
}

/* Convert C double** matrix to Python list of lists */
PyObject* convert_to_python_list(double** c_matrix, int rows, int cols) {
    PyObject* py_list;
    PyObject* py_row;
    int i, j;

    py_list = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        py_row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(c_matrix[i][j]));
        }
        PyList_SetItem(py_list, i, py_row);
    }
    return py_list;
}

/* --- Wrapper Functions --- */

/* Wrapper for sym (1.1) */
static PyObject* wrap_sym(PyObject* self, PyObject* args) {
    PyObject* py_points;
    int n, d;
    double **c_points, **result;
    PyObject* py_result;

    if (!PyArg_ParseTuple(args, "Oii", &py_points, &n, &d)) {
        return NULL; /* Error in parsing arguments */
    }

    c_points = convert_to_c_matrix(py_points, n, d);
    result = sym(c_points, n, d);
    py_result = convert_to_python_list(result, n, n);

    free_matrix(c_points, n);
    free_matrix(result, n);

    return py_result;
}

/* Wrapper for ddg (1.2) */
static PyObject* wrap_ddg(PyObject* self, PyObject* args) {
    PyObject* py_points;
    int n, d;
    double **c_points, **A, **result;
    PyObject* py_result;

    if (!PyArg_ParseTuple(args, "Oii", &py_points, &n, &d)) {
        return NULL;
    }

    c_points = convert_to_c_matrix(py_points, n, d);
    A = sym(c_points, n, d);
    result = ddg(A, n);
    py_result = convert_to_python_list(result, n, n);

    free_matrix(c_points, n);
    free_matrix(A, n);
    free_matrix(result, n);

    return py_result;
}