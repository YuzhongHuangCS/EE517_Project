import datetime
import functools
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os.path
import requests
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

BASE_URL = 'https://56cf3370d8dd3.streamlock.net:1935/live/usc-hecuba.stream/'
DRIVE_SERVICE = None
PATH_ID_DB = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def get_playlist():
    req = requests.get(BASE_URL + 'playlist.m3u8')
    last_line = req.text.split('\n')[-2]
    if re.match(r'chunklist_.*m3u8', last_line):
        return last_line
    else:
        raise NotImplementedError('Unexpected response from playlist', req.text)


def get_chunklist(playlist):
    req = requests.get(BASE_URL + playlist)
    chunklist = []
    timelist = []
    for line in req.text.split('\n'):
        if re.match(r'media_.*.ts', line):
            chunklist.append(line)
        else:
            m = re.search(r'#EXTINF:([0-9]*\.?[0-9]+)', line)
            if m:
                timelist.append(float(m.groups()[0]))

    return chunklist, timelist


@functools.lru_cache(maxsize=128)
def download_media(filename):
    req = requests.get(BASE_URL + filename)
    if req.status_code == 200:
        now = datetime.datetime.now()
        filename_temp = 'temp/' + filename.replace('.ts', '.mpeg')

        with open(filename_temp, 'wb') as f:
            f.write(req.content)
        print('Downloaded', filename_temp)

        path = now.strftime('Hecuba_Camera/%Y/%m/%d/%H/%M')
        filename_drive = now.strftime('%Y_%m_%d_%H_%M_%S_') + filename.replace('.ts', '.mpeg')

        parent = get_parent_by_path(path)
        EXECUTOR.submit(upload_file, filename_temp, filename_drive, parent)
    else:
        raise RuntimeError('Unexpected status', req.status_code)


def get_parent_by_path(current_path, parent_path=None):
    if current_path in PATH_ID_DB:
        print('Fully cached', current_path)
        return PATH_ID_DB[current_path]

    path_parts = current_path.split('/')
    first_child = path_parts[0]
    remaining = path_parts[1:]

    if parent_path is None:
        processed_path = first_child
    else:
        processed_path = parent_path + '/' + first_child

    if processed_path in PATH_ID_DB:
        if len(remaining) > 0:
            print('Partially cached', processed_path)
            return get_parent_by_path('/'.join(remaining), processed_path)
        else:
            print('This should never happened, logic error')

    query = "mimeType='application/vnd.google-apps.folder' and name='{}'".format(first_child)
    if parent_path is not None:
        parent_id = PATH_ID_DB[parent_path]
        query += " and '{}' in parents".format(parent_id)
    else:
        parent_id = None

    response = DRIVE_SERVICE.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])
    if len(files) <= 1:
        if len(files) == 0:
            file_metadata = {
                'name': first_child,
                'mimeType': 'application/vnd.google-apps.folder',
            }
            if parent_id is not None:
                file_metadata['parents'] = [parent_id]

            first_child_id = DRIVE_SERVICE.files().create(body=file_metadata, fields='id').execute().get('id')
            print('Create a new folder', first_child, first_child_id)
        else:
            first_child_id = files[0]['id']
            print('Found an parent', first_child, first_child_id)

        PATH_ID_DB[processed_path] = first_child_id
        if len(remaining) > 0:
            return get_parent_by_path('/'.join(remaining), processed_path)
        else:
            return first_child_id
    else:
        print('More than one file matched, unexpected', files)


def upload_file(filename_temp, filename_drive, parent):
    file_metadata = {
        'name': filename_drive,
        'parents': [parent]
    }
    media = MediaFileUpload(filename_temp, mimetype='video/mp2t')
    file = DRIVE_SERVICE.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print('Uploaded: {} -> File ID: {}'.format(filename_temp, file.get('id')))
    os.remove(filename_temp)


def initialize_google_api():
    global DRIVE_SERVICE

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', ['https://www.googleapis.com/auth/drive'])
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    DRIVE_SERVICE = build('drive', 'v3', credentials=creds)


if __name__ == "__main__":
    initialize_google_api()

    playlist = get_playlist()
    print('Play list: {}'.format(playlist))

    while True:
        try:
            chunklist, timelist = get_chunklist(playlist)
            print('Chunk list: {}, Time list: {}'.format(chunklist, timelist))

            start_time = time.time()
            for filename in chunklist:
                download_media(filename)
            end_time = time.time()

            time_spend = end_time - start_time
            time_expected = np.random.uniform(timelist[0], timelist[0] + timelist[1])
            print('Time spend: {}, time expected: {}'.format(time_spend, time_expected))
            if time_spend < time_expected:
                time.sleep(time_expected - time_spend)

        except Exception as e:
            print(e)

            print('Server refused connection')
            seconds = np.random.randint(0, 60)
            while True:
                try:
                    print('Sleep for {} seconds'.format(seconds))
                    time.sleep(seconds)
                    playlist = get_playlist()
                    print(playlist)
                    break
                except Exception as e2:
                    print(e2)
                    seconds += np.random.randint(0, 60)
