import argparse
import io
import tokenize
from typing import Union
import ast
import numpy as np

parser = argparse.ArgumentParser(description='find Levenshtein distance')
parser.add_argument('indir', type=str, help='Program path file')
parser.add_argument('outdir', type=str, help='Output file')
args = parser.parse_args()


def lev_distance(text1: str, text2: str) -> float:
    '''
    Считает расстояние левенштейна для двух строковых последовательностей с нормировкой на длину текста
    Возвращает коэфициент подобия текстов

    :param text1: str, Первая строковая последовательность
    :param text2: str, Вторая строковая последовательность
    :return: float, 1 - (Нормираванное на длинну расстояние Левенштейна)
    '''
    if text1 == text2:
        return 1
    else:
        _len_file1 = len(text1)
        _len_file2 = len(text2)
        X = np.zeros((_len_file1 + 1, _len_file2 + 1), dtype='int64')
        X[0] = np.arange(_len_file2 + 1)
        X[:, 0] = np.arange((_len_file1 + 1))
        for i in np.arange(X.shape[0] - 1):
            for j in np.arange(X.shape[1] - 1):
                X[i + 1, j + 1] = min(
                    X[i, j] + int(text1[i] != text2[j]),
                    X[i + 1, j] + 1,
                    X[i, j + 1] + 1
                )
        return 1 - (X[-1, -1] / max(_len_file1, _len_file2))


def delete_docs_and_comments(source: str) -> str:
    """
    Удаляет из строковой последовательности комментарии и док-стринги
    :param source: str, Исходный текст для преобразования
    :return: str, Преобразованный текст
    """
    io_obj = io.StringIO(source)
    out = ""
    prev_token_type = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_token_type != tokenize.INDENT:
                if prev_token_type != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_token_type = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(line for line in out.splitlines() if line.strip())
    return out


def prepare_to_compare(text: str) -> str:
    """

    :param text:
    :return:
    """
    file = delete_docs_and_comments(text)
    tree = ast.parse(file)
    normal_tree = str(ast.dump(tree)) \
        .lower().translate({ord(c): None for c in '\'\"[](),= '})
    return normal_tree


def lev_for_py(path1: str, path2: str) -> Union[float, str]:
    """

    :param path1:
    :param path2:
    :return:
    """
    try:
        with open(path1, 'r', encoding="utf8") as f1, open(path2, 'r', encoding="utf8") as f2:
            file1 = f1.read()
            file2 = f2.read()
            text1 = prepare_to_compare(file1)
            text2 = prepare_to_compare(file2)
            return lev_distance(text1, text2)
    except FileNotFoundError as e:
        return f'file {e.filename} not found'


with open(args.indir, 'r') as input_file, open(args.outdir, 'w') as output_file:
    for line in input_file.readlines():
            file1 = line.split(' ')[0].strip()
            file2 = line.split(' ')[1].strip()
            output_file.write(f'{lev_for_py(file1, file2)}\n')
