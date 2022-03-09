import os, pickle


def save():
    file_path = 'utils/farmGPS_roofMaterial.csv'
    with open(file_path, 'r') as f:
        data = f.read().strip().split('\n')

    data.pop(0)
    while ',,,,,' in data:
        data.remove(',,,,,')

    res = []
    for item in data:
        it = item.split(',')
        res.append((it[0], eval(it[1]), eval(it[2]), eval(it[3]), eval(it[4]), eval(it[5])))

    with open('utils/farmGPS_roofMaterial.pkl', 'wb') as fout:
        pickle.dump(res, fout)

def read():
    with open('utils/farmGPS_roofMaterial.pkl', 'rb') as fin:
        res = pickle.load(fin)
        interview_keys, latitudes, longtitudes, zincs, thatchs, house_nums = zip(*res)
        return interview_keys, latitudes, longtitudes, zincs, thatchs, house_nums


if __name__ == '__main__':
    # print(read())
    pass