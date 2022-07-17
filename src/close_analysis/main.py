import os


def parse(filename):
    # township_dict = dict()
    res = {
        'farm_id': [],
        # 'interview_key': [],
        'pmnt_closed': [],
        'ever_closed': [],
        'ever_temp_closed': [],
        'closed': [],
        'reopen': [],
        # 'x_orig': [],
        # 'y_orig': [],
        'x': [],
        'y': [],
        'region': [],
        'intg': [], 
    }
    with open(filename) as f:
        data = f.read().strip().split('\n')
    
    for item in data[1:]:
        item_list = item.split(',')[:13]
    
        res['farm_id'].append(int(item_list[0]))
        res['pmnt_closed'].append(bool(int(item_list[2])))
        res['ever_closed'].append(bool(int(item_list[3])))
        res['ever_temp_closed'].append(bool(int(item_list[4])))
        res['closed'].append(bool(int(item_list[5])))
        res['reopen'].append(bool(int(item_list[6])))
        res['x'].append(float(item_list[10]))
        res['y'].append(float(item_list[9]))
        res['region'].append(item_list[12].strip(' '))
        res['intg'].append(bool(int(item_list[11])))
    
    return res


def get_clusters(filename, edge=0):   # get cluster arrays from a file
    fin = open(filename, 'r')
    data = fin.read().strip().split('cluster')[1:]
    fin.close()
    
    res = {
        'ranges': [], 
        'sizes': [], 
        'farm_ranges': [],
    }
    for i in range(len(data)):
        tmp = data[i].strip().split()
        res['sizes'].append(eval(tmp[0]))
        res['ranges'].append((eval(tmp[1])-edge, eval(tmp[2])-edge, eval(tmp[3])+edge, eval(tmp[4])+edge))
        farm_range = []
        for s in range(res['sizes'][i]):
            farm_range.append((eval(tmp[4*s+5]),eval(tmp[4*s+6]),eval(tmp[4*s+7]),eval(tmp[4*s+8])))
        res['farm_ranges'].append(farm_range)
    
    return res


def get_singles(filename, edge=0):
    fin = open(filename, 'r')
    data = fin.read().strip().split('\n')
    fin.close()
    
    res = {
        'ranges': [],
    }
    for i in range(len(data)):
        tmp = data[i].strip().split()
        res['ranges'].append((eval(tmp[0])-edge, eval(tmp[1])-edge, eval(tmp[2])+edge, eval(tmp[3])+edge))
    
    return res


def get_full_info(interview_info):
    cluster_info_path = '../cluster_detection/result/'
    single_info_path = '../cluster_detection/data/'
    years = range(2019, 2020)
    for year in years:
        cluster_info = get_clusters(cluster_info_path + '{}.txt'.format(year))
        single_info = get_singles(single_info_path + '{}.txt'.format(year))

        for i in range(len(interview_info['farm_id'])):
            flag = False
            for j in range(len(single_info['ranges'])):
                xmin, xmax, ymin, ymax = single_info['ranges'][j]
                x, y = interview_info['x'][i], interview_info['y'][i]
                edge = 0.001
                if x >= xmin - edge and x <= xmax + edge and y >= ymin - edge and y <= ymax + edge:
                    flag = True
                    break
            if flag:
                print('in', end=' ')
            else:
                print('out', end=' ')



if __name__ == '__main__':
    tasks = ['all', 'closed_intg', 'closed']
    files = ['data/' + t + '.csv' for t in tasks]

    for filename in files:
        interview_info = parse(filename)
        full_infos = get_full_info(interview_info)
