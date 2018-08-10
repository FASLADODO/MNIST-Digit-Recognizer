# -*- coding: utf-8 -*-
import os
from dotenv import find_dotenv, load_dotenv
from requests import session
import logging # log intermediate steps that have been completed

# payload for post
payload = {
    'action' : 'login',
    'username' : os.environ.get("KAGGLE_USERNAME"),
    'password' : os.environ.get("KAGGLE_PASSWORD"),
    'rememberme' : 'false'
}

# download from url and store in file_path
def extract_data(login_url, data_url, file_path):
    with session() as c:
        response = c.get(login_url).text
        AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
        payload['__RequestVerificationToken']=AFToken
        c.post(login_url + "?isModal=true&returnUrl=/", data=payload)
        with open(file_path, 'wb') as handle:
            # open url as a stream
            response = c.get(data_url, stream = True)
            # write each block to 1024 bytes
            for block in response.iter_content(1024):
                handle.write(block)
                
# main function, requires the project directory path
def main(project_dir):
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('getting raw data')
    
    # URLs
    loginURL = 'https://www.kaggle.com/account/login'
    trainURL = 'https://www.kaggle.com/c/digit-recognizer/download/train.csv'
    testURL = 'https://www.kaggle.com/c/digit-recognizer/download/test.csv'
    
    # file paths (local)
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')

    # extract data
    extract_data(loginURL, trainURL, train_data_path)
    extract_data(loginURL, testURL, test_data_path)
    logger.info('downloaded raw training and test data')
    
if __name__ == '__main__':
    # getting script file name and append parent directory twice
    # helps to move two levels up since path is /digit_recognizer/src/data
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    # set up logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_fmt)
    
    # find .env automatically by walking up directories until found
    dotenv_path = find_dotenv()
    
    # load up the entries as environment variables
    load_dotenv(dotenv_path)
    
    # call main function
    main(project_dir)