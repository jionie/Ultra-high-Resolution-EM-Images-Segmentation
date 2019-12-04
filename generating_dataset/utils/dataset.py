from PIL import Image
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import numpy as np

class URESDataset(Dataset):
    def __init__(self, data_dir='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE', \
        split=None, csv=None, mode=None, augment=None, size=(1024, 1024)):

        self.data_dir = data_dir
        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment
        self.size = size

        self.uid = list(np.concatenate([np.load(self.data_dir + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(self.data_dir + '/%s'%f).fillna('') for f in csv])

        self.df = df
        self.num_image = len(self.uid)


    def __len__(self):
        return self.num_image


    def __getitem__(self, index):
        image_id, folder = self.uid[index]
        image = cv2.imread(self.data_dir + '/%s/%s.png'%(folder,image_id), cv2.IMREAD_COLOR)

        if self.mode == 'train':
            mask = cv2.imread(self.data_dir + 'label/%s/%s.png'%(folder,image_id), cv2.IMREAD_UNCHANGED)
        else:
            mask = np.zeros((self.size[0], self.size[1], 1), np.uint8)

        image = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=self.size, interpolation=cv2.INTER_LINEAR)
        mask  = mask.astype(np.float32)/255

        if self.augment is None:
            return image, mask
        else:
            return self.augment(image, mask)

def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_mask  = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_mask.append(batch[b][1])

    input = np.stack(input)
    input = input[...,::-1].copy()
    input = input.transpose(0,3,1,2)

    truth_mask = np.stack(truth_mask)
    truth_mask = truth_mask.transpose(0,3,1,2)

    #----
    input = torch.from_numpy(input).float()
    truth_mask = torch.from_numpy(truth_mask).float()


    return input, truth_mask


##############################################################

def tensor_to_image(tensor):
    image = tensor.data.cpu().numpy()
    image = image.transpose(0,2,3,1)
    image = image[...,::-1]
    return image

def tensor_to_mask(tensor):
    mask = tensor.data.cpu().numpy()
    mask = mask.transpose(0,2,3,1)
    return mask


##############################################################

def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = cv2.flip(mask, 1)
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = cv2.flip(mask, 0)

    return image, mask

#
#
# def do_random_crop(image, mask, w, h):
#     height, width = image.shape[:2]
#     x,y=0,0
#     if width>w:
#         x = np.random.choice(width-w)
#     if height>h:
#         y = np.random.choice(height-h)
#     image = image[y:y+h,x:x+w]
#     mask  = mask [y:y+h,x:x+w]
#     return image, mask
#
# def do_random_crop_rescale(image, mask, w, h):
#     height, width = image.shape[:2]
#     x,y=0,0
#     if width>w:
#         x = np.random.choice(width-w)
#     if height>h:
#         y = np.random.choice(height-h)
#     image = image[y:y+h,x:x+w]
#     mask  = mask [y:y+h,x:x+w]
#
#     #---
#     if (w,h)!=(width,height):
#         image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)
#
#     return image, mask
#
# def do_random_crop_rotate_rescale(image, mask, w, h):
#     H,W = image.shape[:2]
#
#     #dangle = np.random.uniform(-2.5, 2.5)
#     #dscale = np.random.uniform(-0.10,0.10,2)
#     dangle = np.random.uniform(-8, 8)
#     dshift = np.random.uniform(-0.1,0.1,2)
#
#     dscale_x = np.random.uniform(-0.00075,0.00075)
#     dscale_y = np.random.uniform(-0.25,0.25)
#
#     cos = np.cos(dangle/180*PI)
#     sin = np.sin(dangle/180*PI)
#     sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
#     tx,ty = dshift*min(H,W)
#
#     src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
#     src = src*[sx,sy]
#     x = (src*[cos,-sin]).sum(1)+W/2
#     y = (src*[sin, cos]).sum(1)+H/2
#     # x = x-x.min()
#     # y = y-y.min()
#     # x = x + (W-x.max())*tx
#     # y = y + (H-y.max())*ty
#
#     if 0:
#         overlay=image.copy()
#         for i in range(4):
#             cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
#         image_show('overlay',overlay)
#         cv2.waitKey(0)
#
#
#     src = np.column_stack([x,y])
#     dst = np.array([[0,0],[w,0],[w,h],[0,h]])
#     s = src.astype(np.float32)
#     d = dst.astype(np.float32)
#     transform = cv2.getPerspectiveTransform(s,d)
#
#     image = cv2.warpPerspective( image, transform, (W, H),
#         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#
#     mask = cv2.warpPerspective( mask, transform, (W, H),
#         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
#
#     return image, mask

# def do_random_log_contast(image, gain=[0.70, 1.30] ):
#     gain = np.random.uniform(gain[0],gain[1],1)
#     inverse = np.random.choice(2,1)
#
#     image = image.astype(np.float32)/255
#     if inverse==0:
#         image = gain*np.log(image+1)
#     else:
#         image = gain*(2**image-1)
#
#     image = np.clip(image*255,0,255).astype(np.uint8)
#     return image
#
# def do_random_noise(image, noise=8):
#     H,W = image.shape[:2]
#     image = image.astype(np.float32)
#     image = image + np.random.uniform(-1,1,(H,W,1))*noise
#     image = np.clip(image,0,255).astype(np.uint8)
#     return image
#
# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# def do_random_contast(image):
#     beta=0
#     alpha=random.uniform(0.5, 2.0)
#
#     image = image.astype(np.float32) * alpha + beta
#     image = np.clip(image,0,255).astype(np.uint8)
#     return image
#
# #----
# ## customize
# def do_random_salt_pepper_noise(image, noise =0.0005):
#     height,width = image.shape[:2]
#     num_salt = int(noise*width*height)
#
#     # Salt mode
#     yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
#     image[tuple(yx)] = [255,255,255]
#
#     # Pepper mode
#     yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
#     image[tuple(yx)] = [0,0,0]
#
#     return image
#
#
#
# def do_random_salt_pepper_line(image, noise =0.0005, length=10):
#     height,width = image.shape[:2]
#     num_salt = int(noise*width*height)
#
#     # Salt mode
#     y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
#     y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
#     for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
#         cv2.line(image,(x0,y0),(x1,y1), (255,255,255), 1)
#
#     # Pepper mode
#     y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
#     y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
#     for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
#         cv2.line(image,(x0,y0),(x1,y1), (0,0,0), 1)
#
#     return image
#
#
# def do_random_cutout(image, mask):
#     height, width = image.shape[:2]
#
#     u0 = [0,1][np.random.choice(2)]
#     u1 = np.random.choice(width)
#
#     if u0 ==0:
#         x0,x1=0,u1
#     if u0 ==1:
#         x0,x1=u1,width
#
#     image[:,x0:x1]=0
#     mask [:,x0:x1]=0
#     return image,mask



# def do_random_special1(image, mask):
#     height, width = image.shape[:2]
#
#     if np.random.rand()>0.5:
#         y = np.random.choice(height)
#         image = np.vstack([image[y:],image[:y]])
#         mask = np.vstack([mask[y:],mask[:y]])
#
#     if np.random.rand()>0.5:
#         x = np.random.choice(width)
#         image = np.hstack([image[:,x:],image[:,:x]])
#         mask = np.hstack([mask[:,x:],mask[:,:x]])
#
#     return image,mask

##############################################################

def run_check_dataset():
    
    dataset = URESDataset(data_dir='/media/jionie/my_disk/Kaggle/URES/input/URES/U-RISC OPEN DATA SIMPLE/U-RISC OPEN DATA SIMPLE', \
        split=['split/train_fold_0_seed_42.npy',], \
        csv=['train.csv'], \
        mode='train', \
        augment=None, \
        size=(1024, 1024)
    )

    print(dataset)
    #exit(0)
    height = 60
    width = 2
    fig, axs = plt.subplots(height, width, figsize=(width * 10, height * 10))

    for n in range(0, len(dataset)):
        i = n #i = np.random.choice(len(dataset))
        
        h = n // width
        w = n % width

        image, label, mask, infor = dataset[i]
        overlay = draw_truth(image, label, mask, infor)

        #----
        print('%05d : %s'%(i, infor.image_id))
        print('label = %s'%str(label))
        print('')
        # #image_show('image',image,0.5)
        # plt.imshow(overlay)
        
        axs[h, w].imshow(overlay) #plot the data
        axs[h, w].axis('off')
        axs[h, w].set_title('%05d : %s'%(i, infor.image_id))
    
    plt.show()




def run_check_dataloader():

    dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_5546.npy',],
        augment = None, #
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 5,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    for t,(input, truth_label, truth_mask, infor) in enumerate(loader):

        print('----t=%d---'%t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth_label', truth_label.shape)
        print('truth_mask ', truth_mask.shape)
        print('')

        if 1:
            batch_size= len(infor)
            
            height = batch_size // 2
            width = 2
            fig, axs = plt.subplots(height, width, figsize=(width * 5, height * 5))

            image       = tensor_to_image(input)
            truth_label = truth_label.data.cpu().numpy()
            truth_mask  = tensor_to_mask(truth_mask)

            for b in range(batch_size):
                
                h = b // width
                w = b % width

                overlay = draw_truth(image[b], truth_label[b], truth_mask[b], infor[b])

                #----
                print('%05d : %s'%(b, infor[b].image_id))
                print('label = %s'%str(truth_label[b]))
                print('')
                axs[h, w].imshow(overlay) #plot the data
                axs[h, w].axis('off')
                axs[h, w].set_title('%05d : %s'%(b, infor[b].image_id))
            
            plt.show()



def run_check_augment():

    def augment(image, label, mask, infor):
        if 1:
            #if np.random.rand()<0.5:  image, mask = do_flip_ud(image, mask)
            if np.random.rand()<0.5:  image, mask = do_flip_lr(image, mask)

        #image, mask = do_random_crop_rescale(image,mask,1600-(256-180),220)
        #image, mask = do_random_crop_rotate_rescale(image,mask,1600-(256-224),224)
        #image = do_random_log_contast(image, gain=[0.70, 1.50])

        #image = do_random_special0(image)

        #image, mask = do_random_special0(image, mask)


        #image, mask = do_random_cutout(image, mask)
        #image = do_random_salt_pepper_line(image, noise =0.0005, length=10)
        #image = do_random_salt_pepper_noise(image, noise =0.005)
        #image = do_random_log_contast(image, gain=[0.50, 1.75])

        # a,b,c = np.random.uniform(-0.25,0.25,3)
        # augment = image.astype(np.float32)/255
        # augment = (augment-0.5+a)*(1+b*2) + (0.5+c)
        # image = np.clip(augment*255,0,255).astype(np.uint8)
        #
        #
        # image = do_random_noise(image, noise=16)

        return image, label, mask, infor


    dataset = CloudDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_5546.npy',],
        augment = None, #
    )
    print(dataset)


    for b in range(len(dataset)):
        
        height = len(dataset) // 2
        width = 2
        fig, axs = plt.subplots(height, width, figsize=(width * 5, height * 5))
        
        image, label, mask, infor = dataset[b]
        overlay = draw_truth(image, label, mask, infor)

        #----
        print('%05d : %s'%(b, infor.image_id))
        print('label = %s'%str(label))
        print('')
        image_show('before',overlay)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, label1, mask1, infor1 =  augment(image.copy(), label.copy(), mask.copy(), infor)
                overlay1 = draw_truth(image1, label1, mask1, infor1)

                #----
                print('%05d : %s'%(b, infor1.image_id))
                print('label = %s'%str(label1))
                print('')
                image_show('after',overlay1)
                cv2.waitKey(0)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_check_dataset()
    #run_check_dataloader()
    #run_check_augment()





