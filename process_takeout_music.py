'''
Google takeout is annoying and saved all music into different 2 GB chunks of songs, with metadata csvs for labelling. 
Want an easy way to organize the music and decide what I want to keep.

The plan:
First move all songs into one folder with subfolders of artist - album
Then go through each album and:
  * Offer to copy, skip, or query, next, back
  * Copy moves to new folder called "Music I want" or somethng
  * Skip ignores (same as next)
  * Query checks the afinfo for each song and presents in df so that it's easy to skip bad quality music
  For each album display the previous and next albums
  * back allows you to go back if you need to

'''
import argparse
import os
import pydub
import re
import shutil
import subprocess
import tempfile

import pandas as pd
import numpy as np

from pydub.playback import play
from tinytag import TinyTag
from tqdm import tqdm

# Load in each meta data csv
# destination = Artist Names - Album Title
# source = Song Title
PARENT_TAKEOUT = '/Users/jadenpieper/Documents/Projects/google-takeout-music/GoogleTakeout'
AUDIOPATH = '/Users/jadenpieper/Documents/Projects/google-takeout-music/songs'
METADATA = os.path.join(AUDIOPATH, 'music-uploads-metadata.csv')
ALBUMPATH = '/Users/jadenpieper/Documents/Projects/google-takeout-music/albums'
SAVEPATH = '/Users/jadenpieper/Documents/Projects/google-takeout-music/selected-albums'

def demosong(songpath, time_s=3):
    song = pydub.AudioSegment.from_mp3(songpath)
    keep_samples = time_s*1000
    out = song[:keep_samples]
    play(out)
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     tmp_path = os.path.join(tmp_dir, 'tmp.wav')
    #     wavfile.write(tmp_path, fs, out)
    #     os.system(f'afplay {tmp_path}')


def gather_songs(parent_path):
    '''
    By default all songs split across multiple folders, move them all to a single folder and generate a metadata csv
    '''
    os.makedirs(AUDIOPATH, exist_ok=False)
    folders = os.listdir(parent_path)
    subpath = os.path.join('YouTube and YouTube Music', 'music-uploads')
    metaname = 'music-uploads-metadata.csv'

    df = pd.DataFrame()
    file_count = 0
    for folder in folders:
        folder_path = os.path.join(parent_path, folder, subpath)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            print(f'{len(files)} in {folder}')
            file_count += len(files)
            for file in tqdm(files):
                filepath = os.path.join(folder_path, file)
                destpath = os.path.join(AUDIOPATH, file)
                shutil.move(filepath, destpath)

def user_album_select(match_df, songpath, duration):
    match_df.index = np.arange(len(match_df))
    while True:
        user_query = f'Select which album you think is the correct match (index number) or pINT to play INT seconds of song.\n{songpath}\n----Duration: {duration:.1f} s\n{match_df}\n'
        user_input = input(user_query)
        if user_input == 'p':
            demosong(songpath)
            continue
        elif 'p' in user_input:
            demo_time = int(user_input.replace('p', ''))
            demosong(songpath, time_s=demo_time)
            continue
        else:
            user_input = int(user_input)
        if user_input < 0 or user_input > len(match_df)-1:
            print(f'Invalid selection of {user_input}')
        else:
            artist = match_df.iloc[user_input]['Artist Names']
            album = match_df.iloc[user_input]['Album Title']
            break
    print(f'Artist Selected: {artist}, album: {album}')
    return artist, album

def song2album(songpath, df):
    tag = TinyTag.get(songpath)
    artist = tag.artist
    album = tag.album
    duration = tag.duration
    duration_tol = 1

    if album is None or artist is None:
        songname, ext = os.path.splitext(os.path.basename(songpath))
        duplicate_pattern = '(\(\d+\)).mp3'
        updated_song = re.sub(duplicate_pattern, '', songname+'.mp3').replace('.mp3', '')
        
        if updated_song != songname:
            # print(f'Looking for {songname} as {updated_song}')
            songname = updated_song
        match_ix = df['Song Title'].str.contains(songname, regex=False)
        if '_' in songname:
            replace_marks = ["'", "&", "?", ":", "/"]
            for mark in replace_marks:
                mark_match = df['Song Title'].str.contains(songname.replace('_', mark), regex=False)
                match_ix = match_ix | mark_match
        
        match_df = df[match_ix]

        # Filter by songs within a decent duration
        duration_match = np.abs(match_df['Duration Seconds'] - duration) < duration_tol
        if any(duration_match):
            match_df = match_df[duration_match]
        
        # Filter out duplicates
        dup_ix = match_df.duplicated(subset=['Song Title', 'Album Title', 'Artist Names'])
        match_df = match_df[~dup_ix]

        nmatches = len(match_df)
        if nmatches == 0:
            print(f'No matches for {songpath}')
        elif nmatches == 1:
            artist = match_df.iloc[0]['Artist Names']
            album = match_df.iloc[0]['Album Title']
        else:
            artist, album = user_album_select(match_df, songpath, duration)
            # import pdb; pdb.set_trace()
    return artist, album
def move_song(songpath, df):
    artist, album = song2album(songpath, df)

    album_folder = f'{artist} - {album}'
    album_path = os.path.join(ALBUMPATH, album_folder)
    os.makedirs(album_path, exist_ok=True)
    dst_path = os.path.join(album_path, os.path.basename(songpath))
    
    do_copy = True
    if os.path.exists(dst_path):
        dst_bitrate = get_bitrate(dst_path)
        src_bitrate = get_bitrate(songpath)
        if dst_bitrate > src_bitrate:
            print(f'Not overwriting {dst_path} with {songpath}, has higher bitrate ({dst_bitrate} > {src_bitrate}')
            do_copy = False
    if do_copy:
        shutil.copy(songpath, dst_path)

def make_albums(audiopath, df):
    songs = os.listdir(audiopath)
    for song in tqdm(songs):
        title, ext = os.path.splitext(song)
        songpath = os.path.join(audiopath, song)
        if ext == '.mp3':
            move_song(songpath, df)
        else:
            print(f'Not mp3: {song}')

# def move_song(row):
#     songname = row['Song Title']+'.mp3'

#     replace_marks = ["'", "&", "?", ":", "/"]
#     for mark in replace_marks:
#         songname = songname.replace(mark, '_')
#     # songname = songname.replace("'", "_")
#     # songname = songname.replace("&", "_")
#     # songname = songname.replace("?", "_")

#     src_path = os.path.join(AUDIOPATH, songname)
#     # tag = TinyTag.get(src_path)
#     album_folder = f"{row['Artist Names']} - {row['Album Title']}"
#     album_path = os.path.join(ALBUMPATH, album_folder)
#     os.makedirs(album_path, exist_ok=True)
#     dst_path = os.path.join(album_path, songname)
#     shutil.copy(src_path, dst_path)

# def make_albums(df):
#     un_ix = []
#     for ix, row in tqdm(df.iterrows()):
#         try:
#             move_song(row)
#         except:
#             # unmatched = pd.concat([unmatched, row])
#             un_ix.append(ix)
#     unmatched = df.iloc[un_ix]
#     unmatched.to_csv('unmatched.csv')
#     print(f'Total unmatched: {len(unmatched)}')

def get_bitrate(filepath):
    cmd = ['afinfo', filepath]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    out = stdout.decode('utf-8')
    srch_str = r'bit rate: (\d+) bits per second'
    match = re.findall(srch_str, out)
    if len(match) > 0:
        bitrate = int(match[0])
    else:
        bitrate = np.nan
    return bitrate
def get_album_bitrate(album_path):
    songs = os.listdir(album_path)
    bitrates = []
    for song in songs:
        songpath = os.path.join(album_path, song)
        bitrate = get_bitrate(songpath)
        bitrates.append(bitrate)
    return bitrates, songs

def album_selection(album_path, output_path, idx=0, start_album=None):
    os.makedirs(output_path, exist_ok=True)
    albums = os.listdir(album_path)
    # for ix, album in enumerate(albums):
    ix = idx
    if start_album is not None:
        start_found = False
    else:
        start_found = True
    n_albums = len(albums)
    while ix < n_albums:
        album = albums[ix]
        if not start_found and album != start_album:
            continue
        else:
            start_found = True
        options_str = 'c - copy, s - skip, n - next, b - back, q - query, e - already copied, x - exit\n'
        print(f'[{ix}/{n_albums}] -- {album}')
        # print(options_str)
        cmd = input(options_str)
        if cmd == 'c':
            src_path = os.path.join(album_path, album)
            dest_path = os.path.join(output_path, album)
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path)
                ix += 1
            else:
                print(f'Already exists {dest_path}')
        elif cmd == 's' or cmd == 'n':
            ix += 1
        elif cmd == 'b':
            if ix > 0:
                ix -= 1
        elif cmd == 'q':
            apath = os.path.join(album_path, album)
            print(apath)
            bitrates, songs = get_album_bitrate(apath)
            adf = pd.DataFrame(dict(song=songs, bitrate=bitrates))
            print(adf)
        elif cmd == 'x':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--make-albums',
        action='store_true',
        help='Parse through song directory to make subdirectories for each album.'
    )

    parser.add_argument(
        '--album-select',
        action='store_true',
        help='Go through copy/query CLI',
    )

    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Starting index for album selection'
    )

    parser.add_argument(
        '--start-album',
        type=str,
        default=None,
        help='Starting album for ablum selection.'
    )

    args = parser.parse_args()

    if not os.path.exists(METADATA):
        gather_songs(PARENT_TAKEOUT)

    df = pd.read_csv(METADATA)
    print(f'Rows in metadata: {len(df)}')
    if args.make_albums:
        make_albums(AUDIOPATH, df)

    if args.album_select:
        album_selection(ALBUMPATH, output_path=SAVEPATH, idx=args.idx, start_album=args.start_album)
    
    # fpath = '/Users/jadenpieper/Documents/Projects/google-takeout-music/albums/All Get Out - Burn Hot The Records/Bring it Home.mp3'
    # fpath = '/Users/jadenpieper/Documents/Projects/google-takeout-music/albums/Jimi Hendrix - Experience Hendrix: The Best Of Jimi Hendrix'
    # print(get_bitrate(fpath))