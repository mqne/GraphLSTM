# generates zip file for uploading to HIM2017 challenge website

from helpers import *
import region_ensemble.model as re

import numpy as np

import os

import zipfile
import tempfile
from tqdm import tqdm


testset_root = r"/mnt/HDD_data/hands2017/data/hand2017_test_0914"
test_list = ["%08d.pkl" % i for i in range(10000, 290001, 10000)] + ["00295510.pkl"]


# get path to npy file
prefix, model_name, epoch = get_prefix_model_name_optionally_epoch()
npyname = predictions_npy_name(model_name, epoch)

# checkpoint_dir = r"/home/matthias-k/GraphLSTM_data/%s" % prefix
# checkpoint_dir += r"/%s" % model_name
# tensorboard_dir = checkpoint_dir + r"/tensorboard/test_him2017"
# prediction_dir = tensorboard_dir

prediction_dir = r"/mnt/HDD_data/data/predictions/test_him2017"
zip_dir = prediction_dir + "/zips"

name = npyname[:-4]
# remove 'predictions_' prefix
if name.startswith("predictions_"):
    name = name[12:]
if prefix not in ("_", None):
    name = prefix + "_" + name

# load predictions
print("Loading predictions …")
predictions = np.load(prediction_dir + "/" + npyname)


# this is duplicated from multiple_hypotheses_extension,
# as importing that module would load tensorflow which is not needed here
HYPOTHESES_AXIS = 1

assert predictions.shape == (295510, 21, 3)
predictions = predictions.reshape(predictions.shape[0], 63)


# # Generate Zip

def test_xyz_and_name_gen(predictions, testset_root, test_list):

    test_image_batch_gen = re.image_batch_generator(testset_root, test_list, re.Const.TEST_BATCH_SIZE)
    # test_params is already loaded

    test_uvd = predictions

    test_param_and_name_gen = re.param_and_name_generator(testset_root, 'tran_para_img', test_list)

    for uvd, param_and_name in zip(test_uvd, test_param_and_name_gen):
        parm = param_and_name[0]
        name = param_and_name[1]
        xyz = re.transform(uvd, parm)
        yield xyz, name


pose_submit_gen = ("frame\\images\\{}\t{}".format(name, '\t'.join(map(str, xyz)))
                   for xyz, name
                   in tqdm(test_xyz_and_name_gen(predictions, testset_root=testset_root, test_list=test_list),
                           total=predictions.shape[0], desc="Conversion to test image coordinates", leave=False,
                           smoothing=0))
pose_submit_array = np.asarray(list(pose_submit_gen))
print("Conversion to test image coordinates done. Gathering results …")

if not os.path.exists(zip_dir):
    os.makedirs(zip_dir)
    print("Created zipfile directory `%s`." % zip_dir)

print("Creating result.txt file in memory …")
temp_file = tempfile.SpooledTemporaryFile()
np.savetxt(temp_file, pose_submit_array, delimiter='\n', fmt="%s")
temp_file.seek(0)

print("Saving zip file …")
with zipfile.ZipFile("%s/%s.zip" % (zip_dir, name), 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("result.txt", temp_file.read())
print("Created zip file %s.zip" % name)
temp_file.close()

print("Done, exiting.")
