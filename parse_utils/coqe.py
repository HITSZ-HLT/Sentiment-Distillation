from utils import append_new_line, save_json
import os, time, json
from datetime import datetime
import ast

class F1Measure:
    def __init__(self):
        self.pred_list = []  # List to store predictions
        self.true_list = []  # List to store ground truths

    def add_predictions(self, idx, preds):
        """Adds a batch of predictions for a specific index."""
        self.pred_list.extend((idx, pred) for pred in preds)

    def add_ground_truths(self, idx, trues):
        """Adds a batch of ground truths for a specific index."""
        self.true_list.extend((idx, true) for true in trues)

    def report(self):
        """Calculates and returns the F1 score."""
        self.f1, self.precision, self.recall = self.calculate_f1()
        return self.f1

    def __getitem__(self, key):
        """Allows retrieval of attributes like a dictionary."""
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{key} is not a valid attribute of F1Measure.")

    def calculate_f1(self):
        """Calculates F1 score along with precision and recall."""



        n_tp = sum(pred in self.true_list for pred in self.pred_list)
        precision = n_tp / len(self.pred_list) if self.pred_list else 1

        n_tp = sum(true in self.pred_list for true in self.true_list)
        recall = n_tp / len(self.true_list) if self.true_list else 1

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return f1, precision, recall



def parse_quintuple_for_true(quintuple_seq, sentence):

    quintuple_tuple = []
    for sub_seq in quintuple_seq:
        new_sub_seq = []
        for item in sub_seq:
            new_sub_seq.append(item.lower())
        quintuple_tuple.append(tuple(new_sub_seq))

    print('true:', quintuple_tuple)
    return quintuple_tuple


def parse_quintuple_for_predict(quintuple_seq, sentence):


    quintuple_seq = quintuple_seq.replace('```python', '')
    quintuple_seq = quintuple_seq.replace('```', '')
    quintuple_seq = quintuple_seq.strip()
    quintuple_seq = quintuple_seq.replace('\n]', ']')
    quintuple_seq = quintuple_seq.replace('[\n', '[')
    quintuple_seq = quintuple_seq.split('\n')[-1]
    quintuple_seq = quintuple_seq.strip()


    valid_flag = True
    print("before parsed :", quintuple_seq)

    try:
        quintuple_seq = ast.literal_eval(quintuple_seq)
        
        
        quintuple_seq = set(quintuple_seq)

        quintuple_seq = [tuple(item) for item in quintuple_seq]
        print("Parsed1 List:", quintuple_seq)
    except:
        try: 
            
            last_paren_index = quintuple_seq.rfind(")")

            if last_paren_index != -1:
                quintuple_seq = quintuple_seq[:last_paren_index+1] + "]"
            else:
                quintuple_seq = quintuple_seq + "]"  # 兜底处理

            # 解析为 Python list
            quintuple_seq = ast.literal_eval(quintuple_seq)
            quintuple_seq = set(quintuple_seq)
            quintuple_seq = [tuple(item) for item in quintuple_seq]
            print("Parsed List2:", quintuple_seq)
        except:
                quintuple_seq = []


    def parse_seq(item):
        if len(item) != 5:
            return False


        subject, object, aspect, opinion, preference = item
        subject = subject.strip().lower()
        object = object.strip().lower()
        aspect = aspect.strip().lower()
        opinion = opinion.strip().lower()
        preference = preference.strip().lower()

        return subject, object, aspect, opinion, preference 


    quintuple_list = []

    for sub_seq in quintuple_seq:
        quintuple = parse_seq(sub_seq)
        if not quintuple:
            valid_flag = False
        else:
            quintuple_list.append(quintuple)

    print('-'*100)
    print('final_list:', quintuple_list)
    print('-'*100)

    return quintuple_seq, quintuple_list, valid_flag


class Result_coqe:
    def __init__(self, data):
        self.data = data

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor > other.monitor

    @classmethod
    def parse_from(cls, outputs):

        data = {}

        ID = 0

        outputs_data = []
        for example, prediction  in outputs:


            sentence = example['sentence']

            coqe_list_true = parse_quintuple_for_true(example['label_seq'], sentence)

            # if coqe_list_true == [('null', 'null', 'null', 'null', 'null')]:
            #     print('all null, skip')
            #     continue

            original_prediction, coqe_list_pred = parse_quintuple_for_predict(prediction,
                                                                                               sentence)[:2]

            # print(coqe_list_pred)
            print('coqe_list_true:', coqe_list_true)
            print('coqe_list_pred:', coqe_list_pred)
            print('sentence:', sentence)
            print('-'*100)


            data[ID] = {
                'ID': example.get('ID', ID),
                'sentence': sentence,
                'prompts':example['prompts'],
                'label_seq':example['label_seq'],
                'golden_label_parsed': coqe_list_true,
                'prediction_parsed': coqe_list_pred,
                'original_output_of_model': original_prediction,
                'dataset': example['dataset']
            }
            ID += 1

            outputs_data.append({
                'sentence': sentence,
                'golden_label_parsed': coqe_list_true,
                'prediction_parsed': coqe_list_pred,
            })

        with open('outputs_data.json', 'w') as f:
            json.dump(outputs_data, f, indent=4)

        return cls(data)

    def cal_metric(self):
        f1 = F1Measure()

        for ID in self.data:
            example = self.data[ID]
            f1.add_ground_truths(ID, example['golden_label_parsed'])
            f1.add_predictions(ID, example['prediction_parsed'])



        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['recall'],
            'precision': f1['precision'],
        }

        self.monitor = self.detailed_metrics['f1']

    def save_prediction(self, output_dir, model_name_or_path, dataset, subname, seed, rank, alpha, target_module, dropout, lr, experiment_name):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        file_name = os.path.join(output_dir, 'result', f'{experiment_name}_{dataset}_{subname}_seed:{seed}_rank:{rank}_alpha:{alpha}_target_module:{target_module}_dropout:{dropout}_lr:{lr}.json')

        print('save prediction to', file_name)
        save_json(
            {
                'data': self.data,
                'meta': (model_name_or_path, subname, dataset, seed, lr, now)
            },
            file_name
        )



    def save_metric(self, output_dir, model_name_or_path, dataset, subname, seed, rank, alpha, target_module, dropout, lr,experiment_name):

        now = datetime.now()
        now = now.strftime("%Y-%m-%d")
        performance_file_name = os.path.join(output_dir, 'performance', 'performance.txt')

        print('save performace to', performance_file_name)
        append_new_line(performance_file_name, json.dumps({
            'experiment_name':experiment_name,
            'metric': self.detailed_metrics['f1'],
            'dataset': dataset,
            'subname': subname,
            'seed': seed,
            'time': time.strftime('%Y-%m-%d %H_%M_%S', time.localtime()),
            'model_name_or_path': model_name_or_path,
            'lr': lr,
            'rank':rank,
            'alpha': alpha,
            'target_module': target_module,
            'dropout': dropout,
            
        }))

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
