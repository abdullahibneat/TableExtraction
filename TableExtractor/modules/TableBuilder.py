from .utils import leafListToDict

def reconstructTable(rows, warped = None, ocrFunction = None):
    # Reconstruct the table following a top-to-bottom approach. Iterate over each row
    # and check for the number of cells. If there are more cells than the previous row,
    # this will indicate columns have been split, and the new row is treated as a new 
    # heading in the table.
    # Otherwise, if the same number of cells appear, this means the row contains new
    # values for the previous column, so add the cell content to the existing column.

    # EXAMPLE TABLE:
    # +-------+-------+
    # |   A   |   B   |
    # +---+---+---+---+
    # | C | D | E | F |
    # +---+---+---+---+
    # | 1 | 2 | 3 | 4 |
    # +---+---+---+---+

    # Store table as a disctionary, where:
    #   key = column name
    #   value = list of cell values
    # In the example above, table will look like the following: 
    #
    # table = {
    #   A: {
    #       C: [1],
    #       D: [2],
    #   },
    #   B: {
    #       E: [3],
    #       F: [4]
    #   }
    # }
    table = {}

    # Columns is a reference to the values of the heading names.
    # For example, in the above example:
    # after the first iteration:    columns = [[], []]
    # after the second iteration:   columns = [[], [], [], []]
    # after the third iteration:    columns = [[1], [2], [3], [4]]
    columns = None
    columns_sizes = [] # Keep track of column sizes

    # Keep track of cell number for use in case OCR fails
    cell_number = 0

    for cells in rows.values():
        cell_sizes = [] # Keep track of cell sizes

        # Extract cell text (will be replaced with OCR)
        cell_contents = []
        for cnt in cells:
            # Extract cell region from image
            x1, y1 = cnt[0]
            x2, y2 = cnt[2]
            cell_sizes.append(x2 - x1) # Add cell width to the list of cell sizes

            # Perform OCR if image and ocrFunction are passed in
            if warped is not None and callable(ocrFunction):
                cell = warped[y1:y2, x1:x2]
                text = ocrFunction(cell)
            # Otherwise return cell number
            else:
                text = str(cell_number)

            if text == "":
                text = "(failed) cell #" + str(cell_number)
            cell_contents.append(text)
            cell_number += 1
        
        if columns is None:
            # FIRST ITERATION
            # Add first row to the table
            for cell in cell_contents:
                table[cell] = []
            columns = list(table.values())
            columns_sizes = cell_sizes

        elif len(cell_contents) > len(columns):
            # DIFFERENT NUMBER OF CELLS
            # Columns have been split, add this row as new headings

            # Replace the previous columns to be new dictionaries.
            # At this line, columns contains the lists of the last headings. Because new
            # headings have been found, the last headings are converted from lists to
            # dictionaries.
            columns = leafListToDict(table)

            # Keep track of the previous headings
            previous_headings = list(columns)

            # Create new columns for each of the new headings
            columns = [[] for _ in cell_contents]

            # Split the new headings into lists of equal size.
            # For instance, in the example table above they are split in groups of 2:
            #   - [C, D] are children of A
            #   - [E, F] are children of B
            # This is done by comparing the current column size (+1%) to the cell sizes.
            # When the column size is exceeded, this will indicate a new column has started.
            current_column_index = 0 # Keep track of current column
            current_width = 0 # Aggregate all cell sizes to check against column size

            for i, heading in enumerate(cell_contents):
                current_width += cell_sizes[i] # Add current cell width to current column

                # If cell doesn't fit in the current column, move to the next one
                if current_width > columns_sizes[current_column_index] * 1.01:
                    current_column_index += 1
                    current_width = cell_sizes[i]

                # Add this new heading as child of previous heading
                previous_headings[current_column_index][heading] = columns[i]

            # Reset columns sizes
            columns_sizes = cell_sizes

        elif len(cell_contents) == len(columns):
            # SAME NUMBER OF CELLS
            # add all cells one by one to each column
            for i in range(len(cells)):
                columns[i].append(cell_contents[i])
        
        elif len(cell_contents) < len(columns):
            # LESS CELLS THAN PREVIOUS ROW
            # This might happen if a table is of the following structure:
            # +-------+-------+
            # |   A   |   B   |
            # +---+---+---+---+
            # | C | D | E | F |
            # +---+---+---+---+
            # |       G       |
            # +---+---+---+---+
            # In this example, the third row has 4 cells, and it's followed by a row
            # of 1 cell. Unfortunately I couldn't find a nice way to store such rows
            # in a JSON file, so the table extraction is stopped at this stage.
            break
    return table
