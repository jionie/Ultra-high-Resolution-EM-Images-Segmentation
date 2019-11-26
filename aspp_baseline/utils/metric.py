from .include import *

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

#----
def probability_mask_to_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = F.adaptive_max_pool2d(probability,1).view(batch_size,-1)
    return probability


#----

def metric_label(probability, truth, threshold=0.5):
    batch_size=len(truth)

    with torch.no_grad():
        probability = probability.view(batch_size,4)
        truth = truth.view(batch_size,4)

        #----
        neg_index = (truth==0).float()
        pos_index = 1-neg_index
        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)

        #----
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        #----
        tn = tn.data.cpu().numpy()
        tp = tp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return tn, tp, num_neg, num_pos


def metric_mask(probability, truth, threshold=0.1, sum_threshold=1):
    
    with torch.no_grad():
        batch_size, num_class, H, W = truth.shape
        probability = probability.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2*(p*t).sum(-1)/((p+t).sum(-1)+1e-12)

        neg_index = (t_sum==0).float()
        pos_index = 1-neg_index

        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)
        dn = (neg_index*d_neg).sum(0)
        dp = (pos_index*d_pos).sum(0)

        #----
        dn = dn.data.cpu().numpy()
        dp = dp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return dn,dp, num_neg,num_pos

def compute_metric( probability_label, probability_mask, truth_label, truth_mask, threshold=(0.60,0.30,1) ):
    
    threshold_label=(np.ones(4)*threshold[0]).reshape(1,4) #0.60
    threshold_mask =(np.ones(4)*threshold[1]).reshape(1,4,1,1) #0.30
    threshold_size =(np.ones(4)*threshold[2]).reshape(1,4) #   1

    num_image = len(truth_label)

    # label
    lp = (probability_label>threshold_label)
    lt = (truth_label>0.5)
    pos_index = (truth_label>0.5)
    neg_index = 1-pos_index

    hit_neg = ((1-lp) == (1-lt))  # True negatives
    hit_pos = (lp == lt)          # True positives
    tn = (hit_neg*neg_index).sum(0)/neg_index.sum(0)
    tp = (hit_pos*pos_index).sum(0)/pos_index.sum(0)

    #----

    mp = (probability_mask>threshold_mask).reshape(num_image,4,-1).astype(np.float32)
    mt = (truth_mask>0.5).reshape(num_image,4,-1).astype(np.float32)

    #only subset
    pos_index = lp * pos_index
    neg_index = lp * neg_index

    dice_neg = (mp.sum(-1) < threshold_size)
    dice_pos = 2*(mp*mt).sum(-1)/((mp+mt).sum(-1)+1e-12)
    dn = (dice_neg*neg_index).sum(0)/neg_index.sum(0)
    dp = (dice_pos*pos_index).sum(0)/pos_index.sum(0)

    #---

    num_truth_pos = (truth_label>0.5).sum(0)
    num_truth_neg = num_image-num_truth_pos
    num_predict_pos = (probability_label>threshold_label).sum(0)
    num_predict_neg = num_image-num_predict_pos


    result=(
       tn, tp, dn, dp, num_truth_neg, num_truth_pos, num_predict_neg, num_predict_pos,
    )
    return result


def summarise_metric(result):
    test_pos_ratio = np.array(
        [NUM_TEST_POS[c][0]/NUM_TEST for c in list(CLASSNAME_TO_CLASSNO.keys())]
    )
    test_neg_ratio = 1-test_pos_ratio


    tn, tp, dn, dp, num_truth_neg, num_truth_pos, num_predict_neg, num_predict_pos = result
    kaggle  = test_neg_ratio*tn + test_neg_ratio*(1-tn)*dn + test_pos_ratio*tp*dp
    kaggle1 = test_neg_ratio*tn + test_pos_ratio*tp


    text  = ''
    text += '                 |   truth  |  predict |              |              |          \n'
    text += '                 | neg  pos | neg  pos | tn     tp    | dn     dp    | kaggle  \n'
    text += '----------------------------------------------------------------------------------------\n'
            # 0      Fish     | 142  158 | 213   87 | 0.958  0.513 | 0.000  0.653 | 0.644  (0.733)
    for c in range(NUM_CLASS):
        text += \
            ' %d  %8s     |'%(c,CLASSNO_TO_CLASSNAME[c]) + \
            ' %3d  %3d |'%(num_truth_neg[c],num_truth_pos[c]) + \
            ' %3d  %3d |'%(num_predict_neg[c],num_predict_pos[c]) + \
            ' %0.3f  %0.3f |'%(tn[c],tp[c]) + \
            ' %0.3f  %0.3f |'%(dn[c],dp[c]) + \
            ' %0.3f (%0.3f)  '%(kaggle[c],kaggle1[c])
        text += '\n'

    text += '\n'
    text += 'kaggle (classification only) = %0.5f (%0.5f)\n'%(kaggle.mean(),kaggle1.mean())
    text += '\n'

    return text



############

# Returns equal error rate (EER) and the corresponding threshold.
def compute_eer(fpr,tpr,threshold):
    fnr = 1-tpr
    abs_diff  = np.abs(fpr-fnr)
    min_index = np.argmin(abs_diff)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, threshold[min_index]



def compute_metric_label(probability_label, truth_label):

    num = len(truth_label)
    t = (truth_label>0.5).astype(np.float32).reshape(-1,NUM_CLASS)
    p = probability_label.reshape(-1,NUM_CLASS)


    #-- AUC/eer -------
    auc=[]
    for c in range(NUM_CLASS):
        fpr, tpr, threshold = sklearn_metrics.roc_curve(t[:,c], p[:,c])
        eer, threshold_eer = compute_eer(fpr,tpr,threshold)
        a = sklearn_metrics.roc_auc_score(t[:,c], p[:,c])
        auc.append([a, eer, threshold_eer])

    #-- tpr/tnr -------
    tnr=[]
    tpr=[]
    for c in range(NUM_CLASS):
        pos = p[:,c][t[:,c]==1]
        neg = p[:,c][t[:,c]==0]
        num_pos = len(pos)
        num_neg = len(neg)

        sort_pos = np.sort(pos)
        sort_neg = np.sort(neg)

        tnr.append((neg<sort_pos[int(0.05*num_pos)]).mean())
        tpr.append((pos>sort_neg[int(0.95*num_neg)]).mean())
    rate_95=(tnr,tpr)

    rate_th={}
    for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
        d = p > th
        tpr = (d*t).sum(0)/t.sum(0)
        tnr = ((1-d)*(1-t)).sum(0)/(1-t).sum(0)
        rate_th[th]=(tnr,tpr)



    return auc, rate_95, rate_th


def summarise_metric_label(result):
    auc, rate_95, rate_th = result

    text  = ''
    text += '** label_metric **\n'


    text += '                     :';
    for c in range(NUM_CLASS): text += '    %d  %8s        |'%(c,CLASSNO_TO_CLASSNAME[c]);
    text +='\n'

    text += '---------------------------------------------------------------------------------------------------------------------------\n'
            #      AUC /  eer@th   :  0.83 / 0.280 @ 0.39  |  0.92 / 0.184 @ 0.58  |  0.81 / 0.276 @ 0.40  |  0.84 / 0.237 @ 0.64  |
            # 0.95-CI   tnr, tpr   :     0.338, 0.532      |     0.598, 0.694      |     0.243, 0.487      |     0.436, 0.357      |

    text += '     AUC /  eer@th   :';
    for c in range(NUM_CLASS): text += '  %0.2f / %0.3f @ %0.2f  |'%(*auc[c],);
    text +='\n'

    text += '0.95-CI   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_95[0][c],rate_95[1][c]);
    text +='\n'

    text += 'th=0.50   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_th[0.50][0][c],rate_th[0.50][0][c],);
    text +='\n'

    text += 'th=0.60   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_th[0.60][0][c],rate_th[0.60][1][c],);
    text +='\n'

    text += 'th=0.70   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_th[0.70][0][c],rate_th[0.70][1][c],);
    text +='\n'

    text += 'th=0.80   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_th[0.80][0][c],rate_th[0.80][1][c],);
    text +='\n'

    text += 'th=0.90   tnr, tpr   :';
    for c in range(NUM_CLASS): text += '     %0.3f, %0.3f      |'%(rate_th[0.90][0][c],rate_th[0.90][1][c],);
    text +='\n'


    return text


############


def compute_metric_mask(predict_label, probability_label, truth_label):
    eps = 1e-15

    num = len(truth_label)

    t = (truth_label>0.5).astype(np.float32).reshape(-1, NUM_CLASS)
    p = probability_label.reshape(-1, NUM_CLASS)




    #-- AUC/eer -------
    auc=[]
    for c in range(NUM_CLASS):
        fpr, tpr, threshold = sklearn_metrics.roc_curve(t[:,c], p[:,c])
        eer, threshold_eer = compute_eer(fpr,tpr,threshold)
        a = sklearn_metrics.roc_auc_score(t[:,c], p[:,c])
        auc.append([a, eer, threshold_eer])

    #-- tpr/tnr -------
    tnr=[]
    tpr=[]
    for c in range(NUM_CLASS):
        pos = p[:,c][t[:,c]==1]
        neg = p[:,c][t[:,c]==0]
        num_pos = len(pos)
        num_neg = len(neg)

        sort_pos = np.sort(pos)
        sort_neg = np.sort(neg)

        tnr.append((neg<sort_pos[int(0.05*num_pos)]).mean())
        tpr.append((pos>sort_neg[int(0.95*num_neg)]).mean())
    rate_95=(tnr,tpr,)

    rate_th=[]
    for th in [0.50, 0.80, 0.90]:
        d = p > th
        tpr = (d*t).sum(0)/t.sum(0)
        tnr = ((1-d)*(1-t)).sum(0)/(1-t).sum(0)
        rate_th.append((tnr,tpr,th))

    d = predict_label
    tpr = (d*t).sum(0)/t.sum(0)
    tnr = ((1-d)*(1-t)).sum(0)/(1-t).sum(0)
    rate=(tnr,tpr,)




    return auc, rate, rate_95, rate_th





def summarise_submission_csv(df):

    df[['image_id','class_name']]= df['Image_Label'].str.split('_', expand = True)
    df['label'] = (df['EncodedPixels']!='').astype(np.int32)
    df_label = pd.pivot_table(df, values = 'label', index=['image_id'], columns = 'class_name').reset_index()

    label = df_label[list(CLASSNAME_TO_CLASSNO.keys())].values
    num_image = len(label)
    pos   = label.sum(0)
    neg   = num_image - pos

    all_zero = (label.sum(1)==0).sum()

    text = ''
    text += 'compare with LB probing ... \n'
    text += '\t\tnum_image = %5d(3698) \n'%num_image
    text += '\n'

    text += '\t\tpos0 = %5d(1864)  %0.3f\n'%(pos[0],pos[0]/1864)
    text += '\t\tpos1 = %5d(1508)  %0.3f\n'%(pos[1],pos[1]/1508)
    text += '\t\tpos2 = %5d(1982)  %0.3f\n'%(pos[2],pos[2]/1982)
    text += '\t\tpos3 = %5d(2382)  %0.3f\n'%(pos[3],pos[3]/2382)
    text += '\n'

    text += '\t\tneg0 = %5d(1834)  %0.3f\n'%(neg[0],neg[0]/1834)
    text += '\t\tneg1 = %5d(1940)  %0.3f\n'%(neg[1],neg[1]/1940)
    text += '\t\tneg2 = %5d(2638)  %0.3f\n'%(neg[2],neg[2]/2638)
    text += '\t\tneg3 = %5d(2017)  %0.3f\n'%(neg[3],neg[3]/2017)
    text += '--------------------------------------------------\n'
    text += '\n'
    text += '\t\tall_zero = %5d (?)\n'%all_zero

    text += '\n'
    return text
