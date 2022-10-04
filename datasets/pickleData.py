import os
import pickle
import pandas as pd


def pickle_data(phase, raf_path):
    df = pd.read_csv(os.path.join(raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
    if phase == 'train':
        dataset = df[df[0].str.startswith('train')]
    else:
        dataset = df[df[0].str.startswith('test')]
    file_names = dataset.iloc[:, 0].values
    # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    labels = dataset.iloc[:, 1].values - 1
    fer_labels = list(labels)
    fer_dict = {}
    fer_images = []
    for i, file_name in enumerate(file_names):
        file_name = file_name.split(".")[0]
        file_name = file_name + "_aligned.jpg"
        file_path = os.path.join(raf_path, 'Image/aligned', file_name)
        fer_images.append(file_path)
    fer_dict['images'] = fer_images
    fer_dict['labels'] = fer_labels
    return fer_dict


if __name__ == '__main__':
    path = ''
    p = 'test'
    data = pickle_data(p, path)
    save_name = './RAF-DB_{}'.format(p)
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)
