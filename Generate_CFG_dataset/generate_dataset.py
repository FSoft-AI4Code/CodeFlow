from __future__ import annotations
import ast, astor, autopep8, tokenize, io, sys
import graphviz as gv
from typing import Dict, List, Tuple, Set, Optional, Type
from cfg import *
import re
import sys
import trace_execution
import os
import io
import linecache
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

output_dir = 'dataset'
inference_dir = 'inference'
os.makedirs(inference_dir, exist_ok=True)
files = os.listdir(output_dir)
files.sort()
out_nodes = []
out_fw_path = []
out_back_path = []
out_exe_path = []
out_file_names = []
max_node = 0
max_edge = 0

for file in files:
    BlockId().counter = 0
    filename = f'./{output_dir}/' + file
    try:
        source = open(filename, 'r').read()
        compile(source, filename, 'exec')
    except:
        print('Error in source code')
        continue
 
    parser = PyParser(source)
    parser.removeCommentsAndDocstrings()
    parser.formatCode()
    try:
        cfg = CFGVisitor().build(filename, ast.parse(parser.script))
    except AttributeError:
        continue
    except IndentationError:
        continue
    except TypeError:
        continue
    except SyntaxError:
        continue
    cfg.clean()
    try:
        cfg.track_execution()
    except Exception:
        continue
    
    code = {}
    for_loop = {}
    for i in cfg.blocks:
        if cfg.blocks[i].for_loop != 0:
            if cfg.blocks[i].for_loop not in for_loop:
                for_loop[cfg.blocks[i].for_loop] = [i]
            else:
                for_loop[cfg.blocks[i].for_loop].append(i)
    first = []
    second = []
    for i in for_loop:
        first.append(for_loop[i][0]+1)
        second.append(for_loop[i][1])
    orin_node = []
    track = {}
    track_for = {}
    for i in cfg.blocks:
        if cfg.blocks[i].stmts_to_code():
            if int(i) == 1:
                st = 'BEGIN'
            elif int(i) == len(cfg.blocks):
                st = 'EXIT'
            else:
                if i in first:
                    line = astor.to_source(cfg.blocks[i].for_name)
                    st = line.split('\n')[0]
                    st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
                else:
                    st = cfg.blocks[i].stmts_to_code()
                    st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
            orin_node.append([i, st, None])
            if st not in track:
                track[st] = [len(orin_node)-1]
            else:
                track[st].append(len(orin_node)-1)
            track_for[i] = len(orin_node)-1
    with open(filename, 'r') as file_open:
        lines = file_open.readlines()
    for i in range(1, len(lines)+1):
        line = lines[i-1]
        #delete \n at the end of each line and delete all spaces
        line = line.strip()
        line = re.sub(r"\s+", "", line).replace('"', "'").replace("(", "").replace(")", "")
        if line.startswith('elif'):
            line = line[2:]
        if line in track:
            orin_node[track[line][0]][2] = i
            if orin_node[track[line][0]][0] in first:
                orin_node[track[line][0]-1][2] = i-0.4
                orin_node[track[line][0]+1][2] = i+0.4
            if len(track[line]) > 1:
                track[line].pop(0)
    for i in second:
        max_val = 0
        for edge in cfg.edges:
            if edge[0] == i:
                if orin_node[track_for[edge[1]]][2] > max_val:
                    max_val = orin_node[track_for[edge[1]]][2]
            if edge[1] == i:
                if orin_node[track_for[edge[0]]][2] > max_val:
                    max_val = orin_node[track_for[edge[0]]][2]
        orin_node[track_for[i]][2] = max_val + 0.5
    orin_node[0][2] = 0
    orin_node[-1][2] = len(lines)+1
    # sort orin_node by the third element
    orin_node.sort(key=lambda x: x[2])
    
    nodes = []
    matching = {}
    for t in orin_node:
        i = t[0]
        if cfg.blocks[i].stmts_to_code():
            if int(i) == 1:
                nodes.append('BEGIN')
            elif int(i) == len(cfg.blocks):
                nodes.append('EXIT')
            else:
                st = cfg.blocks[i].stmts_to_code()
                st_no_space = re.sub(r"\s+", "", st)
                # if start with if or while, delete these keywords
                if st.startswith('if'):
                    st = st[3:]
                elif st.startswith('while'):
                    st = st[6:]
                if cfg.blocks[i].condition:
                    st = 'T '+ st
                if st.endswith('\n'):
                    st = st[:-1]
                if st.endswith(":"):
                    st = st[:-1]
                nodes.append(st)
            matching[i] = len(nodes)
           
    fwd_edges = []
    back_edges = []
    edges = {}
    for edge in cfg.edges:
        if edge not in cfg.back_edges:
            fwd_edges.append((matching[edge[0]], matching[edge[1]]))
        else:
            back_edges.append((matching[edge[0]], matching[edge[1]]))
        if matching[edge[0]] not in edges:
            edges[matching[edge[0]]] = [matching[edge[1]]]
        else:
            edges[matching[edge[0]]].append(matching[edge[1]])
    exe_path = [0 for i in range(len(nodes))]
    for node in cfg.path:
        exe_path[matching[node]-1] = 1
    # check nodes, fwd_edges, back_edges, exe_path not exist in the list and then append
    print(f'Done in {file}')
    if nodes in out_nodes:
        # check the fwd_edges, back_edges, exe_path in the same index
        index = out_nodes.index(nodes)
        if fwd_edges != out_fw_path[index] or back_edges != out_back_path[index] or exe_path != out_exe_path[index]:
            out_nodes.append(nodes)
            out_fw_path.append(fwd_edges)
            out_back_path.append(back_edges)
            out_exe_path.append(exe_path)
            out_file_names.append(filename)
    else:
        out_nodes.append(nodes)
        out_fw_path.append(fwd_edges)
        out_back_path.append(back_edges)
        out_exe_path.append(exe_path)
        out_file_names.append(filename)
 
data = {
    'nodes': out_nodes,
    'forward': out_fw_path,
    'backward': out_back_path,
    'target': out_exe_path,
    'file_name': out_file_names
}
 
df = pd.DataFrame(data)
 
# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
 
# Save train set to CSV
train_df.drop(columns=['file_name'], inplace=True)
train_df.to_csv('train.csv', index=False, quoting=1)
print("Train CSV file has been saved successfully.")
 
# Save test set to CSV
test_file_names = test_df['file_name'].tolist()
test_df.drop(columns=['file_name'], inplace=True)
test_df.to_csv('test.csv', index=False, quoting=1)
print("Test CSV file has been saved successfully.")
 
# Save each test file with the corresponding name in the "inference" folder
for i, filename in enumerate(test_file_names):
    with open(filename, 'r') as f:
        source_code = f.read()
    new_filename = os.path.join(inference_dir, f"code_{i + 2}.py")
    with open(new_filename, 'w') as f:
        f.write(source_code)
print("Test files have been saved successfully.")