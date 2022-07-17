import os

edge = 0

for year in range(2010, 2022):
    fin = open(f'../data/{year}.txt')
    data = fin.read().strip().split('\n')
    fin.close()
    head = '<?xml version="1.0" encoding="utf-8" ?>\n \
<kml xmlns="http://www.opengis.net/kml/2.2">\n \
<Document id="root_doc">\n \
<Folder><name>All</name>\n'
    for i in range(len(data)):
        data[i] = data[i].split()
        jmin, wmin, jmax, wmax = eval(data[i][0]), eval(data[i][1]), eval(data[i][2]), eval(data[i][3])
        jmin = jmin - edge # 0.001
        jmax = jmax + edge # 0.001
        wmin = wmin - edge # 0.001
        wmax = wmax + edge # 0.001
        add = '<Placemark>\n \
<name>{}</name>\n \
<Style><LineStyle><color>ff0000ff</color></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>\n \
<Polygon><outerBoundaryIs><LinearRing><coordinates>{},{} {},{} {},{} {},{} {},{}</coordinates></LinearRing></outerBoundaryIs></Polygon>\n \
</Placemark>' \
            .format('Rect'+str(i), jmin, wmax, jmax, wmax, jmax, wmin, jmin, wmin, jmin, wmax)
        head += add

    head += '</Folder>\n \
</Document></kml>'

    fout = open(f'{year}.kml', 'w')
    fout.write(head)
    fout.close()
  