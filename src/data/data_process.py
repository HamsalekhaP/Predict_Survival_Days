# This code performs the following:
# 1.Fetches segmentation masks and MRI sequences of each patient one by one.
# 2.Finds tumor boundary pixel coordinates, extracts sub-volumes on the fly

# TODO
#  1. Can you optimize?  Runtime O(n^2) and depends on number of contour coordinates
#  2. Can caching help speed up the process

# standard imports
import copy  # module to create copies of objects
import glob  # python module used for path pattern matching
import os  # python module for operating system dependent functionality
from collections import namedtuple # to define tuple blueprint
import random
import math
import csv  # module for reading and writing csv files

# other library imports
import torch
import torch.cuda
import nibabel as nib  # python package that provides read/write access to some medical and neuroimaging file formats
import numpy as np
import scipy.ndimage.measurements as measurements
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# local imports
import config as CONFIG

# Blueprint for saving as tuple all metadata related to data-point. Tuple used for easy unravel of data-point
PatientInfoTuple = namedtuple(
    'PatientInfoTuple',
    'survival_days, max_les_size, patient_uid, age, center_of_mass, mr_type'
)

# Dictionary that collects the max-width(expanse) of tumor along each axis for each of label 1, 2 and 4.
# Each label has array of shape: number_of_data_points x 3.
les_size_arr = {1: [], 2: [], 4: []}


class Mr:
    def __init__(self, patient_id, mr_type):
        base_path = './{}/{}/'.format(CONFIG.NORM_DATASET_PATH, patient_id)
        # data fetched and saved as numpy matrix of shape: 240 x 240 x 155
        mr_data = nib.load(glob.glob("{}{}_{}.nii.gz".format(base_path, patient_id, mr_type))[0]).get_fdata().astype(
            np.float32)
        self.patient_uid = patient_id
        self.mri_values_a = mr_data

    def getRawLesion(self, center_of_mass, width_rck):
        # This function gets the center of mass, and standard width of sub-volume,finds the start and end index of the
        # sub-volume boundary by moving width_rck/2 pixels to the left and right of the center-of-mass along each
        # axis ie., row, column and slice. And then returns the scooped-out sub-volume along with the slice indices.
        slice_list = []
        # along each axis, find the start and end indices wrt center of mass to get the boundaries of lesion volume
        for axis, center_val in enumerate(center_of_mass):
            start_ndx = int(round(center_val - width_rck[axis] / 2))
            end_ndx = int(start_ndx + width_rck[axis])

            if start_ndx < 0:
                print("Start index out of range along {} axis for ID: {}".format(axis, self.patient_uid))
                start_ndx = 0
                end_ndx = int(width_rck[axis])

            if end_ndx > self.mri_values_a.shape[axis]:
                print("End index out of range along {} axis for ID: {}".format(axis, self.patient_uid))
                end_ndx = self.mri_values_a.shape[axis]
                start_ndx = int(self.mri_values_a.shape[axis] - width_rck[axis])

            slice_list.append((start_ndx, end_ndx))

        # This was done just for plotting and verification purpose.
        # Can be refactored to use just slice_list.append(slice(start_ndx, end_ndx))
        slice_create = [slice(ax[0], ax[1]) for ax in slice_list]

        # sub-volume scooped out of MRI sequence
        mr_chunk = self.mri_values_a[tuple(slice_create)]

        return mr_chunk, slice_list


def getLesionSpecs(PTV, seg_type):
    # MR data and segmentation mask have shape(row x column x slice). Slice ranges from-foot to head.
    # This function gets all the pixel coordinates of the requested annotation(label 1,2 or 4), which gives the expanse
    # of the lesion(Planning Target Volume). It then finds maximum width of tumor along each axis by getting the
    # difference between its maximum and minimum coordinate values along each axis. Of the 3 widths takes the max as the
    # max width of the tumor. Return max width and center of mass coordinates.
    PTV_c = copy.copy(PTV)
    PTV_c[PTV_c != seg_type] = 0
    # get pixel coordinates of tumor
    PTVPoints = np.argwhere(PTV == seg_type)
    # find max width and center of mass only if the annotation exists in the segmentation mask of the data sample
    if PTVPoints.size != 0:
        r_axis = PTVPoints[:, 0]
        c_axis = PTVPoints[:, 1]
        k_axis = PTVPoints[:, 2]
        # To find approximate width of lesions. This will facilitate finding a volume that encompasses the tumor.
        les_dim_arr = [max(r_axis) - min(r_axis), max(c_axis) - min(c_axis), max(k_axis) - min(k_axis)]
        # max width of data instance
        max_width = max(les_dim_arr)
        #  to collect data for analysis
        les_size_arr[seg_type].append(les_dim_arr)
        # find center of mass pixel coordinates(r x c x k) for generating lesion volume.
        # List of 3 numbers, one for each coordinate
        com_point = [int(i) for i in measurements.center_of_mass(PTV_c)]
        return max_width, com_point
    else:
        return 0, ''


def getPatientInfoList():
    # this function gets the dataset and creates instances of 'PatientInfoTuple' tuples for each patient.
    # This will be used by the custom dataset loader. In the process,it also consolidates maximum widths,
    # age and survivalDays into one csv file for data-analysis
    patientInfo_list = []
    base_dir = './{}/'.format(CONFIG.NORM_DATASET_PATH)
    # survival_info.csv has survival information
    df = pd.read_csv(base_dir + 'survival_info.csv', usecols=['Brats20ID', 'Survival_days', 'Age'])
    survival_info = dict(
        [(identity, [days, age]) for identity, days, age in zip(df.Brats20ID, df.Survival_days, df.Age)])

    with open(base_dir + 'seg_widths.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Brats20ID", "Survival_days", "Age", "label1", "label2", "label4"])
        # get a list of all patients
        patient_UID = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir + d)]
        for patient in patient_UID:
            # if patient has survival days information, add datapoint details to new csv
            if patient in survival_info.keys():
                PTV = nib.load(os.path.join(base_dir + '/{}/{}'.format(patient, patient + '_seg.nii.gz'))).get_fdata()
                row_list = [patient, survival_info[patient][0], survival_info[patient][1]]
                com_list = []
                # for every segmentation label, get the max-width and center of mass and add to new csv
                for i in CONFIG.SEG_LABELS:
                    max_width, com = getLesionSpecs(PTV, i)
                    row_list.append(max_width)
                    com_list.append(com)
                writer.writerow(row_list)
                is_recur = int(survival_info[patient][0])
                # if every MRI sequence is to be used as a datapoint, add one 'PatientInfoTuple' data for each
                if CONFIG.USE_ALL_MRI_TYPES:
                    for m_type in CONFIG.MRI_TYPES:
                        # if label 1 is not present in the datapoint, add details of label 4 as they are closely related
                        patientInfo_list.append(
                            PatientInfoTuple(is_recur, max_width, patient, survival_info[patient][1],
                                             com_list[2] if com_list[0] == '' else com_list[0], m_type))
                else:
                    patientInfo_list.append(PatientInfoTuple(is_recur, max_width, patient, survival_info[patient][1],
                                                             com_list[2] if com_list[0] == '' else com_list[
                                                                 0], CONFIG.USE_ONE_MRI_TYPE))
    print(len(patientInfo_list))
    patientInfo_list.sort(reverse=True)

    return patientInfo_list


def getMrRawLesion(patient_id, center_of_mass, scoop_size, mr_type):
    # This function return MR sub-volumes
    mr = Mr(patient_id, mr_type)
    mr_channel_data, slice_list = mr.getRawLesion(center_of_mass, scoop_size)
    # if MRI sequences from different modalities need to be stacked, add new axis, get scooped out sub-volumes and stack them
    if CONFIG.USE_AS_CHANNELS:
        mr_channel_data = mr_channel_data[np.newaxis, :, :, :]
        channels = [c for c in CONFIG.MRI_TYPES if c != mr_type]
        for seq in channels:
            mr = Mr(patient_id, seq)
            mr_chunk, slice_list = mr.getRawLesion(center_of_mass, scoop_size)
            mr_channel_data = np.concatenate((mr_channel_data, mr_chunk[np.newaxis, :, :, :]), axis=0)
    return mr_channel_data, slice_list


def getSubVolumeSize():
    # this is the optimal value found after many experiments
    return (54,) * 3


def getSegmentStats():
    colors = ['r', 'b', 'g']
    for seg_label in CONFIG.SEG_LABELS:
        n_les = np.array(les_size_arr[seg_label])
        plt.figure()
        plt.title(f'Distribution of widths for seg label: {seg_label}')
        for axis in range(n_les.shape[1]):
            allWidths = n_les[:, axis]
            plt.hist(allWidths, density=False, color=colors[axis])
            plt.ylabel('Counts')
            plt.xlabel('width in pixels')
        plt.show()
        maxSize = np.amax(les_size_arr[seg_label])
        print('max:', maxSize)
        meanSize = np.mean(les_size_arr[seg_label], axis=0)
        stdSize = np.std(les_size_arr[seg_label], axis=0)
        print(f'mean {meanSize},std dev{stdSize}')
        print(f'2 STD dev away:{meanSize + 2 * stdSize}')


# Code borrowed from https://www.manning.com/books/deep-learning-with-pytorch, Part 2, Chapter 12
def getAugmentedLesion(augmentation_dict, patient_id, center_of_mass, scoop_size, mr_type):
    mr_chunk, slice_list = getMrRawLesion(patient_id, center_of_mass, scoop_size, mr_type)
    if CONFIG.USE_AS_CHANNELS:
        mr_t = torch.tensor(mr_chunk).unsqueeze(0).to(torch.float32)
    else:
        mr_t = torch.tensor(mr_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    # Create a transform matrix and apply all transformations to it
    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict and augmentation_dict['flip']:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict and augmentation_dict['rotate']:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        transform_t @= rotation_t
    # apply the transformation adn retrieve the augmented sub-volume
    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        mr_t.size(),
        align_corners=False
    )
    augmented_chunk = F.grid_sample(
        mr_t,
        affine_t,
        padding_mode='border',
        align_corners=False
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], scoop_size
# borrowed snippet ends here


class MetsDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 augmentation_dict=None,
                 ):
        # create copy to prevent the original list from getting modified when train and validations splits are created
        self.patientInfo_list = copy.copy(getPatientInfoList())
        # uncomment this for visualization
        # getSegmentStats()
        # width of sub-volume
        self.width_rck = getSubVolumeSize()
        self.augmentation_dict = augmentation_dict
        # To facilitate train and validation split when two separate instances of MetsDataset is created
        if isValSet_bool:
            self.patientInfo_list = self.patientInfo_list[::val_stride]
        elif val_stride > 0:
            del self.patientInfo_list[::val_stride]

        random.shuffle(self.patientInfo_list)

        print("{} {} samples".format(len(self.patientInfo_list), "validation" if isValSet_bool else "training"))

    def __len__(self):
        return len(self.patientInfo_list)

    def __getitem__(self, ndx):
        # on the fly get sub-volume and augment it
        patientInfo_tup = self.patientInfo_list[ndx]
        if self.augmentation_dict:
            candidate_lesion_t, slice_list = getAugmentedLesion(self.augmentation_dict,
                                                                patientInfo_tup.patient_uid,
                                                                patientInfo_tup.center_of_mass,
                                                                self.width_rck, patientInfo_tup.mr_type)
        else:
            candidate_lesion_a, slice_list = getMrRawLesion(
                patientInfo_tup.patient_uid,
                patientInfo_tup.center_of_mass,
                self.width_rck, patientInfo_tup.mr_type
            )
            # create tensors to feed to designed model
            candidate_lesion_t = torch.from_numpy(candidate_lesion_a).to(torch.float32)
            if not CONFIG.USE_AS_CHANNELS:
                # This is done to add channel dimension
                candidate_lesion_t = candidate_lesion_t.unsqueeze(0)

        # Regression tensor
        target_t = torch.tensor([patientInfo_tup.survival_days], dtype=torch.float)
        age_t = torch.tensor([patientInfo_tup.age], dtype=torch.float)

        return candidate_lesion_t, target_t, patientInfo_tup.patient_uid, age_t, slice_list


if __name__ == '__main__':
    print(MetsDataset()[0])
