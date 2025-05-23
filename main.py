from dataloader import *
from model import *
from pretrain import *
from utils.utils import *
import train
import os
import time
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer


os.environ['http_proxy'] = 'http://10.16.59.253:7890'
os.environ['https_proxy'] = 'http://10.16.59.253:7890'

parser = ArgumentParser()

parser.add_argument("--dataset", default='teacher', type=str, help="The name of the dataset to train selected.")
parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes.")  # 已知类别比例默认0.75


parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")
parser.add_argument("--data_dir", default='data', type=str,
                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")
parser.add_argument("--save_results_path", type=str, default='outputs', help="The path to save results.")
parser.add_argument("--pretrain_dir", default='pretrain_models', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--train_dir", default='train_models', type=str,
                    help="The output directory where the final model is stored in.")
# parser.add_argument("--bert_model", default="/home/zhanglu/LOOP/model/bert-base-uncased", type=str,
#                     help="The path or name for the pre-trained bert model.")
# parser.add_argument("--tokenizer", default="/home/zhanglu/LOOP/model/bert-base-uncased", type=str,
#                     help="The path or name for the tokenizer")


## bert-base-chinese
parser.add_argument("--bert_model", default="/home/zhanglu/KTN-main/model", type=str,
                    help="The path or name for the pre-trained bert model.")

parser.add_argument("--tokenizer", default="/home/zhanglu/KTN-main/model", type=str,
                    help="The path or name for the tokenizer")


parser.add_argument("--max_seq_length", default=None, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
parser.add_argument("--warmup_proportion", default=0.1, type=float)
parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")
parser.add_argument("--save_model", default=True, type=str, help="Save trained model.")
parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")
parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
parser.add_argument("--rtr_prob", default=0.25, type=float,
                    help="Probability for random token replacement")
parser.add_argument("--labeled_ratio", default=0.1, type=float,
                    help="The ratio of labeled samples in the training set.")
parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")
parser.add_argument("--train_batch_size", default=128, type=int,
                    help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=128, type=int,
                    help="Batch size for evaluation.")
parser.add_argument("--pre_wait_patient", default=20, type=int,
                    help="Patient steps for pre-training Early Stop.")
parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                    help="The pre-training epochs.")
parser.add_argument("--num_train_epochs", default=80, type=float,
                    help="The training epochs.")
parser.add_argument("--lr_pre", default=5e-5, type=float,
                    help="The learning rate for pre-training.")
parser.add_argument("--lr", default=5e-5, type=float,
                    help="The learning rate for training.")
parser.add_argument("--threshold", default=0.5, type=float, help="Value for distinguishing confident novel samples.")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
args = parser.parse_args(args=[])
#tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
data = Data(args)

manager_p = PretrainModelManager(args, data)
manager_p.train(args, data)
# manager_p.load_model(args)
manager_p.evaluation(args, data)

manager = train.Manager(args, data, manager_p.model)
manager.train(args, data)

start_time = time.time()
manager.model.eval()
pred_labels = torch.empty(0, dtype=torch.long).to(manager.device)
total_labels = torch.empty(0, dtype=torch.long).to(manager.device)

for batch in data.test_dataloader:
    batch = tuple(t.to(manager.device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch
    X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
    with torch.no_grad():
         _, logits = manager.model(X, output_hidden_states=True)
    labels = torch.argmax(logits, dim=1)

    pred_labels = torch.cat((pred_labels, labels))  
    total_labels = torch.cat((total_labels, label_ids))  




y_pred = pred_labels.cpu().numpy()
y_true = total_labels.cpu().numpy()

results = clustering_score(y_true, y_pred, data.known_lab)
print('results', results)
end_time = time.time()
print(end_time-start_time)
