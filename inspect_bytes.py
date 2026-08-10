p='main.py'
with open(p,'rb') as f:
    b=f.read()
print('len=',len(b))
print(b[:400])
# print last 200 bytes
print(b[-200:])
