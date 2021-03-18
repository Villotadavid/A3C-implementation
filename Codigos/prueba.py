a =[]

for n in range (0,35):
    if len(a)<20 :
        a.append(None)
        a[n]=n
    else:
        del a[0]
        a.append(n)
print (a)