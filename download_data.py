import os
import shutil
import numpy as np
import pandas as pd
from subprocess import call
from skimage.io import imsave


def convert(ids, X, out_img_folder):
    for id_, x in zip(ids, X):
        imsave('{}/{}.png'.format(out_img_folder, id_), x)


url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
f_name = os.path.basename(url)
if not os.path.exists(f_name):
    print('Downloading {} ...'.format(url))
    call('wget {}'.format(url), shell=True)
f = np.load(f_name)
X_train = f['x_train']
df_train = pd.read_csv(os.path.join('data', 'train.csv'))
df_test = pd.read_csv(os.path.join('data', 'test.csv'))
ids_train = df_train['id'].values
ids_test = df_test['id'].values
ids = np.concatenate([ids_train, ids_test])

imgs_dir = os.path.join('data', 'imgs')
print('Saving images in {}/<id> ...'.format(imgs_dir))
if os.path.exists(imgs_dir):
    shutil.rmtree(imgs_dir)
os.mkdir(imgs_dir)
for id, x in zip(ids, X_train[ids]):
    imsave('{}/{}.png'.format(imgs_dir, id), x)
    # getting rid of extension
    os.rename('{}/{}.png'.format(imgs_dir, id), '{}/{}'.format(imgs_dir, id))
