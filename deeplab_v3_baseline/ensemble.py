import os
from tuils.kaggle import *
from tuils.metric import *
import gc


############
NUM_TRAIN = 5546
NUM_TEST  = 3698

NUM_TEST_POS={ #estimae only !!!! based on public data only !!!
'Fish'   : (1864, 0.5040562466197945),
'Flower' : (1508, 0.4077879935100054),
'Gravel' : (1982, 0.5359653866955111),
'Sugar'  : (2382, 0.6441319632233640), #**
}# total pos:7736  neg:7056

NUM_TRAIN_POS={
'Fish'   : (2765, 0.499),
'Flower' : (3181, 0.574),
'Gravel' : (2607, 0.470),
'Sugar'  : (1795, 0.324),
}

CLASSNAME_TO_CLASSNO = {
'Fish'   : 0,
'Flower' : 1,
'Gravel' : 2,
'Sugar'  : 3,
}

CLASSNO_TO_CLASSNAME = {v: k for k, v in CLASSNAME_TO_CLASSNO.items()}

NUM_CLASS = len(CLASSNAME_TO_CLASSNO)


def run_test_ensemble_segmentation_only():
    
    model = ['deep_se50', 'deep_se101']
    dir = []
    
    for i in range(len(model)):
        dir.append('/media/jionie/my_disk/Kaggle/Cloud/result/deeplab/%s/test/submit/test-tta'%(model[i]))

    out_dir = '/media/jionie/my_disk/Kaggle/Cloud/result/ensemble/'
    
    for m in model:
        out_dir += m + '-'
    out_dir = out_dir[:-1]

    ############################################################
    os.makedirs(out_dir, exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log-ensemble-seg.txt',mode='a')
    
    num_ensemble = 0

    fold = [3, 2]

    for t in range(len(dir)):
        d = dir[t]
        print(t, d)
        
        for i in range(fold[t]):
            num_ensemble += 1
            image_id          = read_list_from_file(d +'/image_id_%s.txt'%(str(i)))
            probability_label = np.load(d +'/probability_label_%s.uint8.npz'%(str(i)))['arr_0']
            probability_mask  = np.load(d +'/probability_mask_%s.uint8.npz'%(str(i)))['arr_0']
            probability_label = probability_label.astype(np.float32) /255
            probability_mask  = probability_mask.astype(np.float32) /255
            
            if (t == 0 and i == 0):
                ensemble_label = probability_label
                ensemble_mask  = probability_mask
            else:
                ensemble_label += probability_label
                ensemble_mask  += probability_mask

    del probability_label, probability_mask
    gc.collect()

    print("num ensemble:", num_ensemble)
    ensemble_label = ensemble_label/num_ensemble
    ensemble_mask  = ensemble_mask/num_ensemble
    ensemble_label = (ensemble_label*255).astype(np.uint8)
    ensemble_mask = (ensemble_mask*255).astype(np.uint8)

    
    write_list_to_file (out_dir + '/image_id.txt', image_id)
    np.savez_compressed(out_dir + '/probability_label.uint8.npz', ensemble_label)
    np.savez_compressed(out_dir + '/probability_mask.uint8.npz', ensemble_mask)


    #---
    threshold_label      = [ 0.70, 0.70, 0.70, 0.70,]
    threshold_mask_pixel = [ 0.30, 0.30, 0.30, 0.30,]
    threshold_mask_size  = [ 1,  1,  1,  1,]


    # threshold_label      = [ 0.825, 1.00, 0.525, 0.50,]
    # threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]
    # threshold_mask_size  = [ 1,  1,  1,  1,]
    #---

    
    csv_file = out_dir +'/ensemble.csv'

    log.write('test submission .... @ ensmble\n')
    print('')
    log.write('threshold_label = %s\n'%str(threshold_label))
    log.write('threshold_mask_pixel = %s\n'%str(threshold_mask_pixel))
    log.write('threshold_mask_size  = %s\n'%str(threshold_mask_size))
    log.write('\n')

    predict_label = ensemble_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
    predict_mask  = ensemble_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

    #-----
    image_id_class_id = []
    encoded_pixel = []
    for b in range(len(image_id)):
        for c in range(NUM_CLASS):
            image_id_class_id.append(image_id[b]+'_%s'%(CLASSNO_TO_CLASSNAME[c]))

            if predict_label[b,c]==0:
                rle=''
            else:
                rle = run_length_encode(predict_mask[b,c])
            encoded_pixel.append(rle)

    df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['Image_Label', 'EncodedPixels'])
    df.to_csv(csv_file, index=False)
    #-----


    ## print statistics ----
    text = summarise_submission_csv(df)
    log.write('\n')
    log.write('%s'%(text))

    ##evalue based on probing results
    # text = do_local_submit(image_id, predict_label,predict_mask=None)
    # log.write('\n')
    # log.write('%s'%(text))


    #--
    # local_result = find_local_threshold(image_id, probability_label, cutoff=[100,0,575,110])
    # threshold_label = [local_result[0][0],local_result[1][0],local_result[2][0],local_result[3][0]]
    # log.write('test threshold_label=%s\n'%str(threshold_label))
    #
    # predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
    # text = do_local_submit(image_id, predict_label,predict_mask=None)
    # log.write('\n')
    log.write('%s'%(text))


# main #################################################################
if __name__ == '__main__':
    #print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_test_ensemble_segmentation_only()

