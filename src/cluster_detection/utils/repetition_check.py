gate = 0.2
f = open('repetition_matrix.csv', 'r')
data = f.read().strip().split()
fn = []
x = [[] for i in range(7)]
for i in range(len(data)):
    lst = data[i].split(',')
    fn.append(lst[0])
    x[0].append(eval(lst[1]))
    x[1].append(eval(lst[2]))
    x[2].append(eval(lst[3]))
    x[3].append(eval(lst[4]))
    x[4].append(eval(lst[5]))
    x[5].append(eval(lst[6]))
    x[6].append(eval(lst[7]))
prev, year = 0, 0
print(year + 2010)
while year < 7:
    cnt = 0
    for i in range(len(fn)):
        cnt += (x[prev][i] == x[year][i])
    print(prev, year, cnt, len(fn))
    if cnt < gate * len(fn):
        prev = year
        print(year + 2010)
    year += 1