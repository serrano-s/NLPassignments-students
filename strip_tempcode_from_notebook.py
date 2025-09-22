from sys import argv
import json


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


def strip_tempcode(name_of_notebook):
    with open(name_of_notebook, 'r') as f:
        contents = json.load(f)
    cell_list = contents['cells']
    for i in range(len(cell_list) - 1, -1, -1):
        cur_cell = cell_list[i]
        if cur_cell['cell_type'] != 'code':
            continue
        source_lines = cur_cell['source']

        commentlines = get_non_empty_commentlines_before_first_codeline(source_lines)
        for line in commentlines:
            line = line.strip().strip('#').strip().upper()
            if line == 'REMOVE FROM STUDENT VERSION':
                del cell_list[i]
                break
    with open(name_of_notebook[:name_of_notebook.rfind('.')] + '-stripped.ipynb', 'w') as f:
        json.dump(contents, f)


if __name__ == '__main__':
    strip_tempcode(argv[1])
