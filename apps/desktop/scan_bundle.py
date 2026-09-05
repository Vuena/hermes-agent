import struct, json, re

p = r'release/win-unpacked/resources/app.asar'
data = open(p, 'rb').read()
jstart = data.find(b'{"files"')
depth = 0
jend = None
for i in range(jstart, min(jstart + 400000, len(data))):
    if data[i:i+1] == b'{':
        depth += 1
    elif data[i:i+1] == b'}':
        depth -= 1
    if depth == 0:
        jend = i
        break
hdr = json.loads(data[jstart:jend+1].decode('utf-8'))
files = hdr['files']

# Walk and print ALL entries with their offsets (even empty dirs)
def walk(d, path='', depth=0):
    for k, v in d.items():
        if k == 'files':
            continue
        p2 = path + '/' + k
        if 'files' in v:
            print('  ' * depth + '[DIR] ' + p2)
            yield from walk(v['files'], p2, depth+1)
        elif 'offset' in v:
            off = int(v['offset'])
            ln = int(v['size'])
            print('  ' * depth + f'[FILE] {p2} off={off} size={ln}')
            yield p2, data[jend+1+off:jend+1+off+ln]

items = list(walk(files))
print('total:', len(items))
