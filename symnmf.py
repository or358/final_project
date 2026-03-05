import sys
import numpy as np
import symnmf  # This imports our compiled C module!

def read_data(filename):
    """Reads the input text file and returns a list of lists."""
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # Ignore empty lines
                    point = [float(x) for x in line.split(',')]
                    data.append(point)
        return data
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

def print_matrix(matrix):
    """Prints a matrix formatted to 4 decimal places, comma-separated."""
    for row in matrix:
        formatted_row = [f"{val:.4f}" for val in row]
        print(",".join(formatted_row))

def init_h(W, n, k):
    """Initializes matrix H based on section 1.4.1 of the assignment."""
    # 1. Calculate m (average of all entries in W)
    m = np.mean(W)
    
    # 2. Set the upper bound for the random interval
    upper_bound = 2 * np.sqrt(m / k)
    
    # 3. Randomly initialize H
    np.random.seed(1234) # Required by the assignment
    H = np.random.uniform(low=0.0, high=upper_bound, size=(n, k))
    
    return H.tolist()

def main():
    # --- 1. Argument Parsing ---
    args = sys.argv
    if len(args) != 4:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        k = int(args[1])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)
        
    goal = args[2]
    filename = args[3]
    
    # --- 2. Read Data ---
    data = read_data(filename)
    n = len(data)
    
    # Optional logical validation:
    if k >= n and goal == 'symnmf':
        print("An Error Has Occurred")
        sys.exit(1)

    # --- 3. Execute the requested goal ---
    try:
        if goal == 'sym':
            # Call our C function to get similarity matrix
            result = symnmf.sym(data, n, len(data[0]))
            print_matrix(result)
            
        elif goal == 'ddg':
            # Call our C function to get diagonal degree matrix
            result = symnmf.ddg(data, n, len(data[0]))
            print_matrix(result)
            
        elif goal == 'norm':
            # Call our C function to get normalized similarity matrix
            result = symnmf.norm(data, n, len(data[0]))
            print_matrix(result)
            
        elif goal == 'symnmf':
            # Full algorithm:
            # 1. Get W from C
            W = symnmf.norm(data, n, len(data[0]))
            
            # 2. Initialize H in Python
            H_init = init_h(W, n, k)
            
            # 3. Pass W and H_init to C for optimization
            final_H = symnmf.symnmf(W, H_init, n, k)
            
            # 4. Print the final H
            print_matrix(final_H)
            
        else:
            # If goal is none of the above
            print("An Error Has Occurred")
            sys.exit(1)
            
    except Exception:
        # Catch any errors coming from the C module
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()