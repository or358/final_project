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

/* Wrapper for norm (1.3) */
static PyObject* wrap_norm(PyObject* self, PyObject* args) {
    PyObject* py_points;
    int n, d;
    double **c_points, **A, **D, **W;
    PyObject* py_result;

    /* Get data from Python */
    if (!PyArg_ParseTuple(args, "Oii", &py_points, &n, &d)) {
        return NULL;
    }

    /* Convert and run C logic */
    c_points = convert_to_c_matrix(py_points, n, d);
    A = sym(c_points, n, d);
    D = ddg(A, n);
    W = norm(A, D, n);
    
    /* Convert back to Python */
    py_result = convert_to_python_list(W, n, n);

    /* Free memory */
    free_matrix(c_points, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free_matrix(W, n);

    return py_result;
}

/* Wrapper for symnmf (1.4) */
static PyObject* wrap_symnmf(PyObject* self, PyObject* args) {
    PyObject *py_W, *py_H;
    int n, k;
    double **c_W, **c_H, **final_H;
    PyObject* py_result;

    /* Get W and H matrices from Python */
    if (!PyArg_ParseTuple(args, "OOii", &py_W, &py_H, &n, &k)) {
        return NULL;
    }

    c_W = convert_to_c_matrix(py_W, n, n);
    c_H = convert_to_c_matrix(py_H, n, k);
    
    /* Run the optimization algorithm */
    final_H = symnmf_optimize(c_W, c_H, n, k);
    
    py_result = convert_to_python_list(final_H, n, k);

    /* Free memory (final_H and c_H point to the same memory location) */
    free_matrix(c_W, n);
    free_matrix(final_H, n);

    return py_result;
}

/* --- Module Definitions --- */

/* 1. Map Python method names to our C wrapper functions */
static PyMethodDef symnmf_methods[] = {
    {"sym", wrap_sym, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", wrap_ddg, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", wrap_norm, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"symnmf", wrap_symnmf, METH_VARARGS, "Optimize H matrix"},
    {NULL, NULL, 0, NULL}
};

/* 2. Define the module structure */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    symnmf_methods
};

/* 3. Initialize the module (this is what Python calls first) */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}