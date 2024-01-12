from sklearn.model_selection import train_test_split
import scipy.io as sci
import numpy as np
import torch
import torch.utils.data as Data
# get data
def getCharacter_data(data_path='../data/Character_imagine/character_imagine_1-process_10-26.mat'):
    # get all data and label
    character_data = sci.loadmat(data_path)
    data = character_data['data']
    label = character_data['label']
    label[:] = label[:] - 1
    label = np.reshape(label, (-1, 1, 1))
    # for solit stratify
    temp_label = label.squeeze(2)
    # NaN to 0
    data = np.nan_to_num(data)
    # split train_data and test_data
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=0, stratify=temp_label, shuffle=True)
    # numpy to Tensor
    train_data_tensor = torch.Tensor(train_data)
    train_labels_tensor = torch.LongTensor(train_label.astype(np.uint8))
    test_data_tensor = torch.Tensor(test_data)
    test_labels_tensor = torch.LongTensor(test_label.astype(np.uint8))

    return train_data_tensor, train_labels_tensor, test_data_tensor, test_labels_tensor

# data package
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_outputs[idx]
