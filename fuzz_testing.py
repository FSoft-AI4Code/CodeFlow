import os
import anthropic
import ast, astor
from cfg import *
import re
import sys
import trace_execution
import os
import io
import pandas as pd
from torchtext import data
from torchtext.data import Iterator
import pandas as pd
import torch
import model
import time
import config
import argparse

def generate_prompt(method_code, feedback=""):
    prompt = f"""
\n\nHuman: You are a terminal. Analyze the following Python code and generate likely inputs for all variables that might raise errors. Add these generated inputs at the beginning of the code snippet.

Example:
Python Method:
if(S[0]=="A" and S[2,-1].count("C")==1):
    cnt=0
    for i in S:
        if(97<=ord(i) and ord(i)<=122):
            cnt+=1
    if(cnt==2):
            print("AC")
    else :
            print("WA")
else :
    print("WA")

Generated Input:
S = 'AtCoder'

Task:
Given the following Python method, generate likely inputs for variables:
{feedback}

Python Method:
{method_code}

Generated Input:
(No explanation needed, only one Generated Input:)
\n\nAssistant:
    """
    return prompt

def get_generated_inputs(claude_api_key, model, method_code, feedback=""):
    client = anthropic.Anthropic(api_key=claude_api_key)
    prompt = generate_prompt(method_code, feedback)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
                {"role": "user", "content": prompt}
            ]
    )
    return response.content[0].text

def add_generated_inputs_to_code(code, inputs):
    lines = code.split('\n')
    # Find the first non-import line
    insert_index = 0
    for i, line in enumerate(lines):
        if not line.startswith(('import', 'from', '\n')):
            insert_index = i
            break
    
    # Insert generated inputs at the found index
    for input_line in inputs.split('\n'):
        if input_line.startswith("Generated"):
            continue
        if input_line.strip():
            lines.insert(insert_index, input_line)
            insert_index += 1

    return '\n'.join(lines)

def read_data(data_path, fields):
    csv_data = pd.read_csv(data_path, chunksize=100)
    all_examples = []
    for n, chunk in enumerate(csv_data):
        examples = chunk.apply(lambda r: data.Example.fromlist([eval(r['nodes']), eval(r['forward']), eval(r['backward']),
                                                                eval(r['target'])], fields), axis=1)
        all_examples.extend(list(examples))
    return all_examples

opt = config.parse()
if opt.claude_api_key == None:
    raise Exception("Lack of CLAUDE api")
if opt.cuda_num == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(f"cuda:{opt.cuda_num}" if torch.cuda.is_available() else "cpu")

TEXT = data.Field(tokenize=lambda x: x.split()[:512])
NODE = data.NestedField(TEXT, preprocessing=lambda x: x[:100], include_lengths=True)
ROW = data.Field(pad_token=1.0, use_vocab=False,
                    preprocessing=lambda x: [1, 1] if any(i > 100 for i in x) else x)
EDGE = data.NestedField(ROW)
TARGET = data.Field(use_vocab=False, preprocessing=lambda x: x[:100], pad_token=0, batch_first=True)

fields = [("nodes", NODE), ("forward", EDGE), ("backward", EDGE), ("target", TARGET)]

print('Read data...')
examples = read_data(f'data/FixEval_complete_train.csv', fields)
train = data.Dataset(examples, fields)
NODE.build_vocab(train, max_size=100000)

orin_nodes = ['BEGIN', "_in = ['2', 3]", 'cont_str = _in[0] * _in[1]', 'cont_num = int(cont_str)', 'sqrt_flag = False', 'p1 = 0', 'p1 < len(range(4, 100))', 'T i = range(4, 100)[p1]', 'sqrt_flag', 'sqrt = i * i', 'cont_num == sqrt', 'T sqrt_flag = True', 'p1 += 1', "T print('Yes')", "print('No')", 'EXIT']
orin_fwd_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (7, 9), (8, 10), (10, 11), (11, 12), (12, 9), (9, 14), (9, 15), (11, 13), (14, 16), (15, 16)]
orin_back_edges = [(13, 7)]
orin_exe_path = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]

net = model.CodeFlow(opt).to(device)
checkpoint_path = f"checkpoints/checkpoints_{opt.checkpoint}/epoch-{opt.epoch}.pt"
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.eval()

outpath = 'fuzz_testing_output'
if not os.path.exists(outpath):
    os.makedirs(outpath)

def extract_inputs(generated_text):
    # Use regular expression to match lines that are variable assignments
    input_lines = re.findall(r'^\s*\w+\s*=\s*.+$', generated_text, re.MULTILINE)
    return '\n'.join(input_lines)

error_dict = {}
locate = 0
for root, _, files in os.walk(opt.folder_path):
    files = sorted(files, key=lambda x: int(x.split('.')[0][5:]))
    for file in files:
        print(f'Fuzz testing file {file}')
        feedback_list = []
        start_time = time.time()
        time_limit = opt.time  # time limit in seconds
        repeat = True
        while repeat:
            if time.time() - start_time > time_limit:
                print(f'Time limit exceeded for file {file}')
                break
            feedback = f"\nThese inputs did not raise runtime errors, avoid to generate the same:\n{feedback_list}" if feedback_list else ""
            file_path = os.path.join(opt.folder_path, file)
            with open(file_path, 'r') as f:
                code = f.read()
                generated_inputs = get_generated_inputs(opt.claude_api_key, opt.model, code, feedback)
                generated_inputs = extract_inputs(generated_inputs)
                print(generated_inputs)
                # Add generated inputs to the original code
                modified_code = add_generated_inputs_to_code(code, generated_inputs)
                filename = os.path.join(outpath, file)
                with open(filename, 'w') as modified_file:
                    modified_file.write(modified_code)

                BlockId().counter = 0
                try:
                    source = open(filename, 'r').read()
                    compile(source, filename, 'exec')
                except:
                    print('Error in source code')
                    exit(1)
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
                    print("Generated input is not valid")
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
                for i in cfg.blocks:
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
                for i in range(len(cfg.path)):
                    if cfg.path[i] == 1:
                        exe_path[matching[i+1]-1] = 1 
                out_nodes=[nodes, orin_nodes]
                out_fw_path=[fwd_edges, orin_fwd_edges]
                out_back_path=[back_edges, orin_back_edges]
                out_exe_path=[exe_path, orin_exe_path]
                data_example = {
                    'nodes': out_nodes,
                    'forward': out_fw_path,
                    'backward': out_back_path,
                    'target': out_exe_path,
                }

                df = pd.DataFrame(data_example)
                # Save to CSV
                df.to_csv(f'{outpath}/output.csv', index=False, quoting=1)
                examples = read_data(f'{outpath}/output.csv', fields)
                test = data.Dataset(examples, fields)
                test_iter = Iterator(test, batch_size=2, device=device, train=False,
                        sort=False, sort_key=lambda x: len(x.nodes), sort_within_batch=False, repeat=False, shuffle=False)
                with torch.no_grad():
                    for batch in test_iter:
                        x, edges, target = batch.nodes, (batch.forward, batch.backward), batch.target.float()
                        if isinstance(x, tuple):
                            pred = net(x[0], edges, x[1], x[2])
                        else:
                            pred = net(x, edges)
                        pred = pred[0].squeeze()
                        pred = (pred > opt.beta).float()
                        if pred[len(nodes)-1] == 1:
                            print("No Runtime Error")
                            feedback_list.append(generated_inputs)
                        else:
                            mask_pred = pred[:len(nodes)] == 1
                            indices_pred = torch.nonzero(mask_pred).flatten()
                            farthest_pred = indices_pred.max().item()
                            error_line = nodes[farthest_pred]
                            print(f"Runtime Error in line: {error_line}")

                            mask_target = target[0][:len(nodes)] == 1                         
                            indices_target = torch.nonzero(mask_target).flatten()
                            farthest_target = indices_target.max().item()
                            true_error_line = nodes[farthest_target]
                            error_dict[file] = [error_line, true_error_line]

                            if farthest_pred == farthest_target:
                                locate += 1
                            repeat = False

locate_true = locate/len(error_dict)*100
print(f'Fuzz testing within {opt.time}s')
print(f'Sucessfully detect: {len(error_dict)}/{len(files)}')
print(f'Bug Localization Acc: {locate_true:.2f}%')
print(error_dict)
