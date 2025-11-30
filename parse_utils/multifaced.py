from utils import append_new_line, save_json
import os, time, json
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import Counter
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




class Result_multifaced:
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
        for example, prediction in outputs:
            sentence = example['sentence']
            # print(prediction)
            extraction_index = prediction.find("Label:")
            if extraction_index == -1:
                prediction = ''
            else:
                prediction = prediction[(extraction_index + len("Label:")):]
                prediction = prediction.strip()
            
            # print('-'*100)
            # print(prediction)
            # print('-'*100)


            if example['dataset'] == 'multifaced/irony18':
                if prediction != 'non-irony' and prediction != 'irony':
                    prediction_processed = 'non-irony'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/hateval':
                if prediction != 'non-hate' and prediction != 'hate':
                    prediction_processed = 'non-hate'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/compsent19':
                if prediction != 'better' and prediction != 'worse' and prediction != 'none':
                    prediction_processed = 'none'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/offenseval':
                if prediction != 'non-offensive' and prediction != 'offensive':
                    prediction_processed = 'non-offensive'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/tweeteval':
                if prediction != 'anger' and prediction != 'joy' and prediction != 'optimism' and prediction != 'sadness':
                    prediction_processed = 'anger'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/pstance':
                if prediction != 'against' and prediction != 'favor':
                    prediction_processed = 'favor'
                else:
                    prediction_processed = prediction
            elif example['dataset'] == 'multifaced/intimacy':
                if prediction not in ["not intimate", "slightly intimate", "moderately intimate", "highly intimate"]:
                    prediction_processed = 'not intimate'
                    print('error')
                else:
                    prediction_processed = prediction
            else:
                prediction_processed = prediction
                
            data[ID] = {
                'ID': example.get('ID', ID),
                'sentence': sentence,
                'prompts': example['prompts'],
                'golden_label': example['label_seq'],
                'prediction_processed': prediction_processed,
                'original_output_of_model': prediction,
                'dataset': example['dataset']
            }
            ID += 1

        return cls(data)

    def cal_metric(self):

        ground_truths = []
        predictions = []

        dataset = self.data[0]['dataset']
        for ID in self.data:
            example = self.data[ID]
            ground_truths.append(example['golden_label'])
            predictions.append(example['prediction_processed'])
        
        # print('ground_truths:', ground_truths)
        print(Counter(ground_truths))

        # print('predictions:', predictions)
        print(Counter(predictions))

        macro_f1 = f1_score(ground_truths, predictions, average='macro')
        macro_precision = precision_score(ground_truths, predictions, average='macro')
        macro_recall = recall_score(ground_truths, predictions, average='macro')
        
        
    
        self.detailed_metrics = {
            'f1': macro_f1,
            'recall': macro_precision,
            'precision': macro_recall,
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
