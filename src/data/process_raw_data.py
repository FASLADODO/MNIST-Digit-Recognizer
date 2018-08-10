import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging # log intermediate steps that have been completed

                
# main function, requires the project directory path
def main(project_dir):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('processing raw data')
    
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    labeled_file_path = os.path.join(raw_data_path, 'train.csv')
    unlabeled_file_path = os.path.join(raw_data_path, 'test.csv')

    # read the data with all default parameters
    labeled_df = pd.read_csv(labeled_file_path, index_col = None)
    unlabeled_df = pd.read_csv(unlabeled_file_path, index_col = None)
    logger.info('raw data has been retrieved')
    
    # drop columns with only a single value
    for c in labeled_df.columns:
        if labeled_df[c].nunique() == 1:
            labeled_df.drop([c], axis = 1, inplace = True)
            unlabeled_df.drop([c], axis = 1, inplace = True)
    logger.info('variance zero columns dropped')
    
    # rescale columns from 0-255 to 0-1
    labeled_df.iloc[:,1:] = labeled_df.iloc[:,1:].applymap(lambda x : x / 255)
    unlabeled_df.iloc[:,:] = unlabeled_df.iloc[:,:].applymap(lambda x : x / 255)
    logger.info('columns scaled to 0-1')
    
    # add ImageId column, reorder so it's first
    labeled_df['ImageId'] = labeled_df.index + 1
    columns = [column for column in labeled_df.columns if column != 'ImageId']
    columns = ['ImageId'] + columns
    labeled_df = labeled_df[columns]

    unlabeled_df['ImageId'] = unlabeled_df.index + 1
    columns = [column for column in unlabeled_df.columns if column != 'ImageId']
    columns = ['ImageId'] + columns
    unlabeled_df = unlabeled_df[columns]
    logger.info('ImageId column added to the front')
    
    # 80-20 train-test split, stratify split based on 'label' column
    train_df, test_df = train_test_split(labeled_df,
                                         test_size = 0.2,
                                         random_state = 42,
                                         stratify = labeled_df.label)
    logger.info('performed 50-50 train-test split')
    
    # define file paths for processed data
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_train_processed_path = os.path.join(processed_data_path, 'train_processed.csv')
    write_test_processed_path = os.path.join(processed_data_path, 'test_processed.csv')
    write_unlabeled_processed_path = os.path.join(processed_data_path, 'unlabeled_processed.csv')
    
    # write processed data to files
    train_df.to_csv(write_train_processed_path, index = False)
    test_df.to_csv(write_test_processed_path, index = False)
    unlabeled_df.to_csv(write_unlabeled_processed_path, index = False)
    logger.info('processed data written to files')
    
    
    
if __name__ == '__main__':
    # getting script file name and append parent directory twice
    # helps to move two levels up since path is /digit_recognizer/src/data
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    # set up logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_fmt)
    
    # call main function
    main(project_dir)