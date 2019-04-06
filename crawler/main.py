import datetime
import functools
import os
import re
import time

import numpy as np
import requests

BASE_URL = 'https://56cf3370d8dd3.streamlock.net:1935/live/usc-hecuba.stream/'


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
                time = m.groups()[0]
                timelist.append(float(time))

    return chunklist, timelist


@functools.lru_cache(maxsize=128)
def download_media(filename):
    req = requests.get(BASE_URL + filename)
    if req.status_code == 200:
        now = datetime.datetime.now()
        folder = now.strftime('data/%Y/%m/%d/%H/%M/')
        os.makedirs(folder, exist_ok=True)
        filename_mpeg = folder + now.strftime('%Y_%m_%d_%H_%M_%S_') + filename.replace('.ts', '.mpeg')
        with open(filename_mpeg, 'wb') as f:
            f.write(req.content)
        print('Done', filename_mpeg)
    else:
        raise RuntimeError('Unexpected status', req.status_code)


if __name__ == "__main__":
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

            playlist = get_playlist()
            print(playlist)
