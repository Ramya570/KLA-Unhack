import pandas as pd
import numpy as np

def read_input_txt(input_txt_path):
    with open(input_txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract ORIENTATION and DIESIZE
    orientation = None
    die_rows = die_cols = None

    for line in lines:
        if line.startswith("ORIENTATION:"):
            try:
                orientation = int(line.split(":")[1].strip())
            except:
                raise ValueError("ORIENTATION is not a valid integer!")

        elif line.startswith("DIESIZE:"):
            try:
                size_str = line.split(":")[1].strip()
                die_cols, die_rows = map(int, size_str.split(","))  # Note: order is cols, rows
            except:
                raise ValueError("DIESIZE format is invalid!")

    if orientation is None or die_rows is None or die_cols is None:
        raise ValueError("ORIENTATION or DIESIZE missing in INPUT.txt")

    # Extract ink matrix
    matrix_lines = []
    for line in lines:
        if line.startswith("ROWDATA:"):
            row_data = line.split("ROWDATA:")[1].strip()
            parts = row_data.split()
            if all(p.isdigit() for p in parts):
                matrix_lines.append([int(p) for p in parts])

    if not matrix_lines:
        raise ValueError("No matrix data found in INPUT.txt")

    ink_matrix = np.array(matrix_lines)

    return orientation, die_rows, die_cols, ink_matrix


def rotate_to_0(ink_matrix, orientation):
    if orientation == 0:
        return ink_matrix
    elif orientation == 90:
        return np.rot90(ink_matrix, -1)
    elif orientation == 180:
        return np.rot90(ink_matrix, 2)
    elif orientation == 270:
        return np.rot90(ink_matrix, 1)
    else:
        raise ValueError(" Invalid orientation. Must be 0, 90, 180, or 270.")


def process_die_analysis(matrix_csv_path, input_txt_path, output_path):
    # Load wafer
    wafer_df = pd.read_csv(matrix_csv_path, header=None)
    wafer_matrix = wafer_df.values.astype(int)

    # Load ink + info
    orientation, die_rows, die_cols, ink_matrix = read_input_txt(input_txt_path)

    # Rotate ink matrix
    ink_matrix = rotate_to_0(ink_matrix, orientation)

    # Determine the actual output size from the ink matrix after rotation
    output_rows = ink_matrix.shape[0]
    output_cols = ink_matrix.shape[1]

    # Ensure wafer matrix matches the ink matrix size
    wafer_matrix = wafer_matrix[:output_rows, :output_cols]
    if wafer_matrix.shape[0] < output_rows or wafer_matrix.shape[1] < output_cols:
        padded_wafer = np.zeros((output_rows, output_cols), dtype=int)
        padded_wafer[:wafer_matrix.shape[0], :wafer_matrix.shape[1]] = wafer_matrix
        wafer_matrix = padded_wafer

    final_matrix = np.zeros((output_rows, output_cols), dtype=int)
    good = 0
    bad = 0

    for i in range(output_rows):
        for j in range(output_cols):
            if wafer_matrix[i][j] == 0:
                final_matrix[i][j] = 0
            elif ink_matrix[i][j] == 100:
                final_matrix[i][j] = 100
                good += 1
            elif ink_matrix[i][j] == 1:
                final_matrix[i][j] = 1
                bad += 1
            else:
                final_matrix[i][j] = 0

    # Write output
    with open(output_path, 'w') as f:
        f.write(f"NO OF GOOD DIES:{good}\n")
        f.write(f"NO OF BAD DIES:{bad}\n")
        for row in final_matrix:
            row_str = " ".join(f"{val:03}" for val in row)
            f.write(f"ROWDATA:{row_str}\n")

    print(f"Output saved to: {output_path}")


# ==== RUN SECTION ====
matrix_file = "C:/Users/sweet/Downloads/Students/Dataset/M1/T8/MATRIX.csv"
input_file = "C:/Users/sweet/Downloads/Students/Dataset/M1/T8/INPUT.txt"
output_file = "C:/Users/sweet/Downloads/Students/Output/M1/M1T8.txt"

process_die_analysis(matrix_file, input_file, output_file)
