import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
        
        
def get_input_stats(DATASET):
    if DATASET == 'CIFAR10':
        data_mean, data_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    elif DATASET == 'CIFAR100':
        data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)

    return data_mean, data_std

def reshape_output(model_output):
    batch_size = len(model_output)
    return model_output.reshape(batch_size,-1)

class KNN():
    def __init__(self, model, k, device, transformer = False):
        super(KNN, self).__init__()
        self.k = k
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.transformer = transformer

    def extract_features(self, loader):
        x_lst = []
        features = []
        label_lst = []
        print(f'Preprocessing loader...(len({len(loader)}))')
        with torch.no_grad():
            for input_tensor, label in loader:
                if self.transformer:
                    h = reshape_output(self.model.extract_feat(input_tensor.to(self.device))[0])
                else:
                    h = reshape_output(self.model(input_tensor.to(self.device))[0])
                features.append(h)
                x_lst.append(input_tensor)
                label_lst.append(label)

            x_total = torch.stack(x_lst)
            h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

            return x_total, h_total, label_total

    def knn(self, features, labels, k=1):
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features_np = features.cpu().view(-1, feature_dim).numpy()
            labels_np = labels.cpu().view(-1).numpy()
            # fit
            self.cls = KNeighborsClassifier(k, metric='cosine').fit(features_np, labels_np)
            acc = self.eval(features, labels)

        return acc

    def eval(self, features, labels):
        feature_dim = features.shape[-1]
        features = features.cpu().view(-1, feature_dim).numpy()
        labels = labels.cpu().view(-1).numpy()
        acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
        
        return acc

    def _find_best_indices(self, h_query, h_ref):
        h_query = h_query / h_query.norm(dim=1).view(-1, 1)
        h_ref = h_ref / h_ref.norm(dim=1).view(-1, 1)
        scores = torch.matmul(h_query, h_ref.t())  # [query_bs, ref_bs]
        score, indices = scores.topk(1, dim=1)  # select top k best
        return score, indices

    def fit(self, train_loader, test_loader=None):
        with torch.no_grad():
            x_train, h_train, l_train = self.extract_features(train_loader)
            print('Evaluate on train data...')
            train_acc = self.knn(h_train, l_train, k=self.k)
            test_acc = 'no test data'
            
            if test_loader is not None:
                x_test, h_test, l_test = self.extract_features(test_loader)
                print('Evaluate on test data...')
                test_acc = self.eval(h_test, l_test)
            
            return train_acc, test_acc


def knn_eval(model, train_loader, val_loader, device, transformer):
    ssl_evaluator = KNN(model=model, k=1, device='cuda', transformer=transformer)
    train_acc, val_acc = ssl_evaluator.fit(train_loader, val_loader)
    print(f'Knn eval: train_acc={train_acc:.2f} val_acc={val_acc:.2f}')
    return train_acc, val_acc