import numpy as np
import pandas as pd
from skimage.morphology import label

# Run-length encoding
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def process_prediction(preds_test_upsampled,test_ids):
	new_test_ids = []
	rles = []
	for n, id_ in enumerate(test_ids):
	    rle = list(prob_to_rles(preds_test_upsampled[n]))
	    rles.extend(rle)
	    new_test_ids.extend([id_] * len(rle))
		
	# Create submission DataFrame
	sub = pd.DataFrame()
	sub['ImageId'] = new_test_ids
	sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	sub.to_csv('sub-dsbowl2018-1.csv', index=False)