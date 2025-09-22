from sys import argv
import json
from strip_tempcode_from_notebook import strip_tempcode


name_of_notebook = argv[1]
strip_tempcode(name_of_notebook)
name_of_notebook_stripped = name_of_notebook[:name_of_notebook.rfind('.')] + '-stripped.ipynb'


with open(name_of_notebook_stripped, 'r') as f:
    contents = json.load(f)


def get_non_empty_commentlines_before_first_codeline(cell_lines):
    commentlines = []
    for line in cell_lines:
        line_contents = line.strip()
        if line_contents == '':
            continue
        elif line_contents.startswith('#'):
            commentlines.append(line_contents)
        else:
            break
    return commentlines


def get_all_nonempty_lines_after_lineind(lineind, cell_lines):
    nonempty_lines = []
    return_this = False
    for line in cell_lines[lineind + 1:]:
        line_contents = line.strip()
        if line_contents == '':
            if len(nonempty_lines) > 0:
                nonempty_lines.append(line_contents)
        elif line_contents.startswith('#'):
            nonempty_lines.append(line_contents)
        else:
            nonempty_lines.append(line_contents)
            return_this = True
    if return_this:
        return nonempty_lines
    else:
        return []


cell_list = contents['cells']
for i in range(len(cell_list) - 1, -1, -1):
    # leave NO_AUTOGRADE alone. Just looking for BEGIN/END SOLUTION
    cur_cell = cell_list[i]
    if cur_cell['cell_type'] != 'code':
        continue
    # and delete all cell outputs
    cur_cell["outputs"] = []

    char_before_function_def = None
    num_of_that_char_before_function_def = -1
    source_lines = cur_cell['source']
    for line in source_lines:
        if char_before_function_def == None and num_of_that_char_before_function_def == -1:
            if line.lstrip().startswith("def "):
                to_count = line[:line.index('def ')]
                if to_count.count(' ') == len(to_count):
                    num_of_that_char_before_function_def = len(to_count)
                    char_before_function_def = ' '
                    break
                elif to_count.count('\t') == len(to_count):
                    num_of_that_char_before_function_def = len(to_count)
                    char_before_function_def = '\t'
                    break
                else:
                    assert False

    currently_replacing = False
    replaced_with = None
    for i in range(len(source_lines) - 1, -1, -1):
        line = source_lines[i]
        if (not currently_replacing) and line.strip().strip('#').strip().upper() == 'END SOLUTION':
            currently_replacing = True
            num_leading_chars = len(line[:line.index('#')])
            if num_leading_chars > num_of_that_char_before_function_def:
                replaced_with = 'raise NotImplementedError'
            else:
                replaced_with = '# YOUR CODE HERE'
            if char_before_function_def is not None:
                line = (char_before_function_def * num_leading_chars) + replaced_with
            else:
                line = (' ' * num_leading_chars) + replaced_with
            source_lines[i] = line
            source_lines.insert(i + 1, '\n')
        elif currently_replacing:
            if line.strip().strip('#').strip().upper() == 'BEGIN SOLUTION':
                currently_replacing = False
                num_leading_chars = len(line[:line.index('#')])
                assert num_leading_chars == len(source_lines[i + 1][:source_lines[i + 1].index(replaced_with)]), (
                    str(source_lines[i: i + 2]))
                source_lines[i] = '\n'
            else:
                del source_lines[i]


with open(name_of_notebook_stripped, 'w') as f:
    json.dump(contents, f)
