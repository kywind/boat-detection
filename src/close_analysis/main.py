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


def getclusters(filename, edge=0):   # get cluster arrays from a file
    fin = open(filename, 'r')
    data = fin.read().strip().split('cluster')[1:]
    fin.close()
    
    ranges, sizes, contents = [], [], []
    for i in range(len(data)):
        tmp = data[i].strip().split()
        sizes.append(eval(tmp[0]))
        ranges.append((eval(tmp[1])-edge, eval(tmp[2])-edge, eval(tmp[3])+edge, eval(tmp[4])+edge))
        content = []
        for s in range(sizes[i]):
            content.append((eval(tmp[4*s+5]),eval(tmp[4*s+6]),eval(tmp[4*s+7]),eval(tmp[4*s+8])))
        contents.append(content)
    
    return ranges, sizes, contents


def get_cluster_info(interview_info, cluster_info_path):
    # with open(cluster_info_path) as f:
    #     cluster_data = f.read().strip().split('\n')
    ranges, sizes, contents = getclusters(cluster_info_path)
    print(ranges, sizes, contents)



if __name__ == '__main__':
    tasks = ['all', 'closed_intg', 'closed']
    files = ['data/' + t + '.csv' for t in tasks]
    cluster_info_path = '../cluster_detection/result/2019.txt'
    
    for filename in files:
        interview_info = parse(filename)
        interview_and_cluster_info = get_cluster_info(interview_info, cluster_info_path)
