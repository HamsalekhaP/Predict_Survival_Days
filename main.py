# standard imports
import argparse
# module provides access to some variables used by the interpreter
import sys

# local imports
# from src.models import train
# from src.models import test
from src.models import reg_train
import config as CONFIG


def main(args):
    print('Working!')
    CONFIG.cli_args = args
    # For BrainMetastases classification problem. Temporarily parked
    # if args.mode == 'Train':
    #     train.Training().main()

    # For Survival days Regression problem
    if args.mode == 'Train':
        reg_train.RegTraining().main()
    else:
        print('Incorrect mode')
        # To use when Test data becomes available
        # test.Test().main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Test model')
    # define cli parameters
    parser.add_argument("--mode",
                        help="Currently only configured to run only --mode=rTrain for training survival Regression "
                             "model.Will enable --mode=train for train mode or --mode=test for test mode when data is "
                             "available",
                        type=str,
                        default="Train"
                        )
    parser.add_argument('--epochs',
                        help='Number of epochs to train for',
                        default=50,
                        type=int,
                        )

    parser.add_argument('--batch_size',
                        help='Enter batch size for train and test modes. --batch_size=64',
                        type=int,
                        default=32
                        )
    parser.add_argument('--num-workers',
                        help='Number of worker processes for background data loading',
                        default=8,
                        type=int)

    parser.add_argument('--tb-prefix',
                        help='Folder name for Tensorboard SummaryWriter run event.Include this to generate a run '
                             'specific folder that will house all visualizations',
                        default='RunFolder1',
                        type=str)

    parser.add_argument('--comment',
                        help='Tensorboard SummaryWriter event file name. This will generate train and validation '
                             'event files with this name. Helps to use comment that represents the run',
                        default='runFile1',
                        type=str)
    # borrowed snippet from https://www.manning.com/books/deep-learning-with-pytorch, Part 2
    parser.add_argument('--augmented',
                        help="True if you want to augment data with all augmentation strategies. Else specify each "
                             "strategy ",
                        default=False,
                        )
    parser.add_argument('--augment_flip',
                        help="Augment data by randomly flipping the data left-right, up-down and front-back.",
                        default=True,
                        )
    parser.add_argument('--augment_offset',
                        help="Augment data by randomly offsetting the data slightly along the X and Y axes.",
                        default=0.1,
                        )
    parser.add_argument('--augment_scale',
                        help="Augment data by randomly increasing or decreasing the size of the lesion.",
                        default=0.2,
                        )
    parser.add_argument('--augment_rotate',
                        help="Augment data by randomly rotating the data around the head-foot axis.",
                        default=15,
                        )
    parser.add_argument('--augment_noise',
                        help="Augment data by randomly adding noise to the data.",
                        default=0.05,
                        )
    # borrowed snippet ends here

    parser.add_argument("--resume",
                        help="Flag to resume training from where it was stopped",
                        type=bool,
                        default=True)

    sys_argv = sys.argv[1:]
    CONFIG.cli_args = parser.parse_args(sys_argv)
    main(CONFIG.cli_args)
