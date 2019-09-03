import tensorflow as tf
from PIL import Image
import PIL
import scipy.misc as misc
import numpy as np

data_root_path="../input0/VOCdevkit_train/VOC2012"


def make_one_hot(x,n):
    '''
    print(x.shape)
    one_hot=np.zeros([x.shape[0],x.shape[1],n])
    print(one_hot.shape)
    for i in range(n):
        #print(x==i)
        print(one_hot[x==i])
        one_hot[x==i][i]=1
    '''
    one_hot = np.zeros([x.shape[0], x.shape[1], n])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            one_hot[i,j,x[i,j]]=1
    return one_hot


class voc_reader:
    def __init__(self,resize_width,resize_height,train_batch_size,val_batch_size):
        self.train_file_name_list=self.load_file_name_list(file_path=data_root_path+"/ImageSets/Segmentation/train.txt")
        self.val_file_name_list=self.load_file_name_list(file_path=data_root_path+"/ImageSets/Segmentation/val.txt")
        self.row_file_path=data_root_path+"/JPEGImages/"
        self.label_file_path=data_root_path+"/SegmentationClass/"
        self.train_batch_index=0
        self.val_batch_index=0
        self.resize_width=resize_width
        self.resize_height=resize_height
        self.n_train_file=len(self.train_file_name_list)
        self.n_val_file=len(self.val_file_name_list)
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        print(self.n_train_file)
        print(self.n_val_file)
        self.n_train_steps_per_epoch=self.n_train_file//self.train_batch_size
        self.n_val_steps_per_epoch=self.n_val_file//self.val_batch_size


    def load_file_name_list(self,file_path):
        file_name_list=[]
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        return file_name_list

    def next_train_batch(self):
        train_imgs=np.zeros((self.train_batch_size,self.resize_height,self.resize_width,3))
        train_labels=np.zeros([self.train_batch_size,self.resize_height,self.resize_width,21])
        if self.train_batch_index>=self.n_train_steps_per_epoch:
            print("next epoch")
            self.train_batch_index=0
        #print('------------------')
        #print(self.train_batch_index)
        for i in range(self.train_batch_size):
            index=self.train_batch_size*self.train_batch_index+i
            #print('index'+str(index))
            img = Image.open(self.row_file_path+self.train_file_name_list[index]+'.jpg')
            img=img.resize((self.resize_height,self.resize_width),Image.NEAREST)
            img=np.array(img)
            train_imgs[i]=img
            #print(img.shape)
            np.set_printoptions(threshold=np.inf)

            label=Image.open(self.label_file_path+self.train_file_name_list[index]+'.png')
            label=label.resize((self.resize_height,self.resize_width),Image.NEAREST)
            label=np.array(label, dtype=np.int32)
            #print(label[label>20])
            #label[label == 255] = -1
            label[label==255]=0
            #print(label)
            #print(label.shape)
            one_hot_label=make_one_hot(label,21)
            train_labels[i]=one_hot_label
            #print(one_hot_label.shape)
            #print(label)
            #print(label)

        self.train_batch_index+=1
        #print('------------------')


        return train_imgs,train_labels


    def next_val_batch(self):
        val_imgs = np.zeros((self.val_batch_size, self.resize_height, self.resize_width, 3))
        val_labels = np.zeros([self.val_batch_size, self.resize_height, self.resize_width, 21])
        if self.val_batch_index>=self.n_val_steps_per_epoch:
            print("next epoch")
            self.val_batch_index=0
        #print('------------------')
        #print(self.val_batch_index)


        for i in range(self.val_batch_size):
            index=self.val_batch_size*self.val_batch_index+i
            #print('index'+str(index))
            img=Image.open(self.row_file_path+self.val_file_name_list[index]+'.jpg')
            img = img.resize((self.resize_height, self.resize_width), Image.NEAREST)
            img = np.array(img)
            val_imgs[i]=img

            label = Image.open(self.label_file_path + self.val_file_name_list[index] + '.png')
            label = label.resize((self.resize_height, self.resize_width), Image.NEAREST)
            label = np.array(label, dtype=np.int32)
            # print(label[label>20])
            # label[label == 255] = -1
            label[label == 255] = 0
            # print(label)
            # print(label.shape)
            one_hot_label = make_one_hot(label, 21)
            val_labels[i]=one_hot_label
        #print('------------------')
        self.val_batch_index+=1



        return val_imgs,val_labels