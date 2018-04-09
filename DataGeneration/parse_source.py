import os
import json
from text_to_image import create_img
from itertools import islice

file_types = {
    'Java':['.java',0],
    'Python':['.py',0],
    'C-Sharp':['.cs',0],
    'Ruby':['.rb',0],
    'Javascript':['.js',0],
    'C':['.c',0],
    'Go':['.go',0],
    'C-Plus-Plus':['.cpp',0],
    'Scala':['.scala',0],
    'MATLAB':['.m',0],
    'R':['.R',0],
    'CSS':['css',0],
    'PHP':['.php',0],
    'Perl':['.pm',0],
    'Shell':['.sh',0]

}

data = {
    'Java':[],
    'Python':[],
    'C-Sharp':[],
    'Ruby':[],
    'Javascript':[],
    'C':[],
    'Go':[],
    'C-Plus-Plus':[],
    'Scala':[],
    'MATLAB':[],
    'R':[],
    'CSS':[],
    'PHP':[],
    'Perl':[],
    'Shell':[]

}
rootdir = 'SourceFiles'

def window(seq, n=15):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[2:] + (elem,)
        yield result

def setup_image_dir():
    for k in file_types.keys():
        if not os.path.exists('Images/'+k):
            os.mkdir('Images/'+k)

setup_image_dir()

last_file_type = None
count = 0
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        lang = subdir.split('/')[1]
        if lang != last_file_type:
            last_file_type = lang
            count = 0
        if f.endswith(file_types[lang][0]):
            with open(os.path.join(subdir,f)) as lang_file:
                lines = lang_file.readlines()
                for patch in window(lines,min(len(lines),15)):
                    if len(patch) < 5:
                        continue
                    code = ''.join(patch)
                    #print code.count('\n')
                    create_img(code,'Images/'+lang+'/'+str(count))
                    data[lang].append('Images/'+lang+'/'+str(count)+'.png')
                    count += 1
                    file_types[lang][1] += 1
            #print lang,f

c = 0
for k in file_types.keys():
    print k,file_types[k][1]
    c += file_types[k][1]

print c

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)
