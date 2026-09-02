import ast


import sys
p='main.py'
try:


    s=open(p,'r',encoding='utf-8').read()


    ast.parse(s)


    print('OK')


except SyntaxError as e:


    print('SyntaxError', e.lineno, e.offset)


    if e.text:


        print('LINE:', e.text.rstrip())


    sys.exit(1)


