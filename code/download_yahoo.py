# Script to download yahoo videos
# Needs ffmpeg and ffprobe on PATH
# - Retrieves and caches M3U8 URLs
# - Downloads videos from M3U8 with ffmpeg
# - Check video durations with ffprobe
# - Deletes potentially corrupted videos

import urllib.request, json
import sys
from urllib.parse import urlparse
import os
import subprocess
import pandas as pd
import numpy as np
import scipy.stats

cwd = os.path.join(os.getcwd(), '..')
dest_folder = os.path.join(cwd, 'video_files')
videos = pd.read_json(os.path.join(cwd, 'videos', 'yahoo_thumbnail_cikm2016.json'))

uuids = videos.uuid.values

def get_m3u8(uuid):
	params = "?protocol=http&format=mpd%2Cfmp4%2Cm3u8%2Cmp4%2Cwebm&srid=1970741294&rt=html&devtype=desktop&offnetwork=false&region=US&site=news&expb=fp-US-en-US-def&expn=y20&lang=en-US&width=808&height=524&resize=true&ps=ncfap795&autoplay=true&image_sizes=130x72&excludePS=true&acctid=145&synd=&pspid=1197618800&plidl=&topic=&pver=7.86.731.1528748180&try=1&failover_count=0&firstVideo=73065f05-26e7-33d1-b708-466a93639358&hlspre=true&env=prod"
	url = r"https://video-api.yql.yahoo.com/v1/video/sapi/streams/" + uuid + params

	with urllib.request.urlopen(url) as fin:
		data = json.loads(fin.read().decode())
		data = data['query']['results']['mediaObj'][0]
		streams = [x for x in data['streams'] if x['format'] == 'm3u8_playlist']
		stream = max(streams, key=lambda s: s['width'])
		m3u8_stream_path = stream['host'] + stream['path']
		return m3u8_stream_path

def get_m3u8_safe(uuid):
	try:
		print(f'Success get_m3u8({uuid})')
		return get_m3u8(uuid)
	except Exception as e:
		print(f'Failed get_m3u8({uuid})')
		return f'FAIL: {str(e)}'
	
def download_command(m3u8_url, destination):
	return f'ffmpeg -i "{m3u8_url}" -bsf:a aac_adtstoasc -vcodec copy -c copy -crf 50 {destination}'

# M3U8 retrieval
n = len(uuids)
m3u8s = []

viable_videos_path = os.path.join(cwd, 'viable_videos.csv')

if not os.path.exists(viable_videos_path):
	print('Downloading m3u8s')
	# Download m3u8s
	for i, x in enumerate(uuids):
		if i % 100 == 0:
			print(f'{i}/{n} processed')
		m3u8 = get_m3u8_safe(x)
		m3u8s.append(m3u8)

	# Filter videos
	viable_m3u8s_mask = [not 'FAIL' in x for x in m3u8s]
	videos['m3u8'] = m3u8s
	viable_videos = videos[viable_m3u8s_mask]
	viable_videos['file_path'] = viable_videos['uuid'].map(lambda x: os.path.join(dest_folder, f'{x}.mp4'))

	viable_videos.to_csv(viable_videos_path)
else:
	print('Getting cached viable videos')
	viable_videos = pd.read_csv(viable_videos_path)

if not os.path.exists(dest_folder):
	os.makedirs(dest_folder)

# Download videos from M3U8
processes = []
vids_to_process = list(viable_videos.iterrows())
parallel_download_max = 24

while len(vids_to_process) > 0:
	i, row = vids_to_process.pop()
	if not os.path.exists(row.file_path):
		while len(processes) > parallel_download_max:
			processes = [p for p in processes if p.poll() is None]
		processes.append(subprocess.Popen(download_command(row.m3u8, row.file_path), shell=True))
	else:
		print(f'{row.uuid} exists')

while len(processes) > 0:
	processes = [p for p in processes if p.poll() is None]

# Check videos
ffprobe_processes = []
vids_to_process_ffprobe = list(viable_videos.iterrows())
vids_to_process_ffprobe_len = len(vids_to_process_ffprobe)
while len(vids_to_process_ffprobe) > 0:
	i, row = vids_to_process_ffprobe.pop(0)
	if i % 100 == 0:
		print(f'ffprobe check: {i}/{vids_to_process_ffprobe_len} processing started')
	
	if os.path.exists(row.file_path):
		p = subprocess.Popen(f'ffprobe -i {row.file_path} -show_entries format=duration -v quiet -of csv="p=0"', shell=True, stdout=subprocess.PIPE)
		ffprobe_processes.append(p)
	else:
		ffprobe_processes.append(None)

ffprobe_results = []
error = False
corrupted_files = []
for i, fp in enumerate(ffprobe_processes):
	file_path = viable_videos.iloc[i].file_path
	print('ffprobe check for:', file_path)
	if fp is None:
		ffprobe_results.append(None)
	else:
		try:
			ffprobe_result = fp.communicate()[0].decode('utf-8').strip()
			ffprobe_results.append(float(ffprobe_result))
		except Exception as e:
			print(e)
			print('Encountered error while checking ffprobe: the file is corrupted and will be deleted')
			corrupted_files.append(file_path)
			error = True
			
if error:
	for fp in [x for x in ffprobe_processes if x is not None]:
		fp.kill()
	for cf in corrupted_files:
		os.remove(cf)
	print('Corrupted files have been cleaned: please restart the script')
	raise Exception

ffprobe_results = np.array(ffprobe_results)

ffprobe_success = len([x for x in ffprobe_results if x is not None])
print(f'{ffprobe_success}/{vids_to_process_ffprobe_len} ffprobe successes')

vids_to_process_ffprobe = list(viable_videos.iterrows())
failed_vids_mask = np.array([x is None for x in ffprobe_results])
success_vids_mask = ~failed_vids_mask
failed_vids = [x[1].file_path for x in np.array(vids_to_process_ffprobe)[failed_vids_mask]]
success_vids = viable_videos[~failed_vids_mask]
print('Files missing:', failed_vids)

durations = np.vstack((success_vids.duration, [int(x) for x in ffprobe_results[success_vids_mask]])).T
durations_delta = durations[:,0]-durations[:,1]

print('Durations delta for available videos', scipy.stats.describe(durations_delta))
duration_delta_threshold = 1
high_duration_delta_mask = abs(durations_delta) > duration_delta_threshold
duration_delta_error_count = len([x for x in high_duration_delta_mask if x])
print(f'Duration deltas above {duration_delta_threshold}: {duration_delta_error_count}/{len(durations_delta)}')

print('Deleting potentially corrupted videos')
potentially_corrupted_videos = success_vids[high_duration_delta_mask]
for i, row in potentially_corrupted_videos.iterrows():
	os.remove(row.file_path)
print('Deleted potentially corrupted videos')