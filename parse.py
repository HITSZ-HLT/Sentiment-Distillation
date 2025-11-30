import json
import numpy as np
import os
from collections import defaultdict

def parse_icl_fsa_matrics(experiments):
    """
    解析性能数据并以美观表格输出，包含平均值列，按指定顺序打印列。

    :param path: 数据文件路径
    :param experiments: 需要解析的实验名称列表
    """
    path = "./output/result.txt"
    # 读取数据
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line))

    # 定义列顺序
    datasets1 = [ 'sc/imdb', 'sc/yelp2', 'sc/sst2', 'sc/twitter', 'multifaced/irony18', 'multifaced/tweeteval', 'multifaced/pstance', 'multifaced/intimacy',]

    datasets2 = [ 'absa/asqp_rest16', 'absa/opener', 'absa/atsa_rest16', 'absa/acsa_rest16', ]

    datasets = datasets1 + datasets2

    header = f"{'Experiment':<100} | " + " | ".join(f"{dataset:<50}" for dataset in datasets) + " | Average"
    print(header)
    # 处理每个实验
    for experiment in experiments:
        result = defaultdict(list)
        filtered_data = [item for item in data if experiment in item[0]]

        for item in filtered_data:
            keys = item[0].split('_')
            model, dataset, seed, k = keys[-10], keys[-9], keys[-8], keys[-7][-1]
            if  'absa' in model:
                dataset = keys[-10] + '_' + keys[-9]
                model = keys[-11]
            # print(model, dataset, seed, k)
            result[(model, dataset, k)].append(item[1]['f1'])


        # 计算每个数据集的平均值
        final_result = {}
        for k, v in result.items():
            # print(k[1], sum(v)/len(v)*100)
            # print(v)
            # if len(v) != 3:
            #     print('error')
            final_result[k[1]] = sum(v) / len(v) * 100
        # print(final_result)
 
        # 计算行平均值
        row_values = [final_result.get(dataset.lower(), 0) for dataset in datasets]
        row_average = sum(row_values) / len(datasets)

        # 打印结果
        row = f"{experiment:<100} | " + " | ".join(f"{value:<30.2f}" for value in row_values) + f" | {row_average:<30.2f}"
        print(row)





experiment_names= [
"qwen2.5-distilled",
"llama-3-1b-distilled",
"llama-3-3b-distilled"
    
]


parse_icl_fsa_matrics(experiment_names)

