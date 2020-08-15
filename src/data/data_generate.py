# This file contains a function that gets each MRI sequence from 'reg_dataset' folder, performs z-score normalization
# and saves it in a normDataset folder

# standard imports

import os  # python module for operating system dependent functionality
import shutil  # module that offer high level operations on files

# other library imports
import numpy as np
import nibabel as nib  # python package that provides read/write access to some medical and neuroimaging file formats


def generateNormalisedData():
    path = './normDataset'
    if os.path.isdir(path):
        print('Removing existing directory...')
        shutil.rmtree(path)
    print('Making Directory...')
    os.makedirs(path)

    base_dir = './{}/'.format('reg_dataset')
    patient_UID = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir + d)]

    print(f'Starting to save {len(patient_UID)} patient data...')

    for ind in range(1, len(patient_UID) + 1):
        patient = f'BraTS20_Training_{ind:03d}'
        os.makedirs('{}/{}/'.format(path, patient))
        print(f'Saving file:{patient}')
        for mri_type in ['t1', 't2', 't1ce', 'flair', 'seg']:
            patient_file = os.path.join(base_dir + '/{}/{}_{}'.format(patient, patient, mri_type + '.nii.gz'))
            dest_file = os.path.join(path + '/{}/{}_{}'.format(patient, patient, mri_type + '.nii.gz'))
            if mri_type == 'seg':
                shutil.copy(patient_file, dest_file)
            else:
                # perform z-score normalization for all but segmentation mask
                mri_nifti = nib.load(patient_file)
                mri_mat = mri_nifti.get_fdata()
                mri_mean = np.mean(mri_mat)
                mri_std = np.std(mri_mat)

                normed = (mri_mat - mri_mean) / (mri_std + 0.000001)
                # along with data matrix, save additional metadata as well
                new_img = nib.Nifti1Image(normed, mri_nifti.affine, mri_nifti.header)
                nib.save(new_img, dest_file)

    shutil.copy(base_dir + '/survival_info.csv', path + '/survival_info.csv')


if __name__ == '__main__':
    generateNormalisedData()
