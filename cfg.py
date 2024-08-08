from __future__ import annotations
import ast, astor, autopep8, tokenize, io, sys
import graphviz as gv
from typing import Dict, List, Tuple, Set, Optional, Type
import re
import sys
import trace_execution
import os
import io
import linecache
import random

# TODO later: graph
'''
1. add a color dictionary for condition calls
2. node shape (may be added into class Block)
'''

class SingletonMeta(type):
    _instance: Optional[BlockId] = None

    def __call__(self) -> BlockId:
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class BlockId(metaclass=SingletonMeta):
    counter: int = 0

    def gen(self) -> int:
        self.counter += 1
        return self.counter


class BasicBlock:

    def __init__(self, bid: int):
        self.bid: int = bid
        self.stmts: List[Type[ast.AST]] = []
        self.calls: List[str] = []
        self.prev: List[int] = []
        self.next: List[int] = []
        self.condition = False
        self.for_loop = 0
        self.for_name = Type[ast.AST]

    def is_empty(self) -> bool:
        return len(self.stmts) == 0

    def has_next(self) -> bool:
        return len(self.next) != 0

    def has_previous(self) -> bool:
        return len(self.prev) != 0

    def remove_from_prev(self, prev_bid: int) -> None:
        if prev_bid in self.prev:
            self.prev.remove(prev_bid)

    def remove_from_next(self, next_bid: int) -> None:
        if next_bid in self.next:
            self.next.remove(next_bid)

    def stmts_to_code(self) -> str:
        code = ''
        for stmt in self.stmts:
            line = astor.to_source(stmt)
            code += line.split('\n')[0] + "\n" if type(stmt) in [ast.If, ast.For, ast.While, ast.FunctionDef,
                                                                 ast.AsyncFunctionDef] else line
        return code

    def calls_to_code(self) -> str:
        return '\n'.join(self.calls)


class CFG:

    def __init__(self, name: str):
        self.name: str = name
        self.filename = name

        # I am sure that in original code variable asynchr is not used
        # And I think list finalblocks is also not used.

        self.start: Optional[BasicBlock] =  None
        self.func_calls: Dict[str, CFG] = {}
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: Dict[Tuple[int, int], Type[ast.AST]] = {}
        self.back_edges: List[Tuple[int, int]] = []
        self.lst_temp : List[Tuple[int, int]] = []
        self.graph: Optional[gv.dot.Digraph] = None
        self.execution_path: List[int] = []
        self.path: List[int] = []
        self.revert: Dict[int, int] = {}
        self.store_True: List[int] = []

    def _traverse(self, block: BasicBlock, visited: Set[int] = set(), calls: bool = True) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            st = block.stmts_to_code()
            if st.startswith('if'):
                st = st[3:]
            elif st.startswith('while'):
                st = st[6:]
            if block.condition:
                st = 'T ' + st
            # Check if the block is in the path and highlight it
            node_attributes = {'shape': 'ellipse'}
            if block.bid in self.path:
                node_attributes['color'] = 'red'
                node_attributes['style'] = 'filled'

            self.graph.node(str(block.bid), label=st, _attributes=node_attributes)

            for next_bid in block.next:
                self._traverse(self.blocks[next_bid], visited, calls=calls)
                self.graph.edge(str(block.bid), str(next_bid))

    def get_revert(self):
        code = {}
        for_loop = {}
        for i in self.blocks:
            if self.blocks[i].for_loop != 0:
                if self.blocks[i].for_loop not in for_loop:
                    for_loop[self.blocks[i].for_loop] = [i]
                else:
                    for_loop[self.blocks[i].for_loop].append(i)
        first = []
        second = []
        for i in for_loop:
            first.append(for_loop[i][0]+1)
            second.append(for_loop[i][1])
        orin_node = []
        track = {}
        track_for = {}
        for i in self.blocks:
            if self.blocks[i].stmts_to_code():
                if int(i) == 1:
                    st = 'BEGIN'
                elif int(i) == len(self.blocks):
                    st = 'EXIT'
                else:
                    if i in first:
                        line = astor.to_source(self.blocks[i].for_name)
                        st = line.split('\n')[0]
                        st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
                    else:
                        st = self.blocks[i].stmts_to_code()
                        st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
                orin_node.append([i, st, None])
                if st not in track:
                    track[st] = [len(orin_node)-1]
                else:
                    track[st].append(len(orin_node)-1)
                track_for[i] = len(orin_node)-1
        with open(self.filename, 'r') as file:
            lines = file.readlines()
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
            for edge in self.edges:
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
        for t in orin_node:
            i = t[0]
            if self.blocks[i].stmts_to_code():
                if int(i) == 1:
                    nodes.append('BEGIN')
                elif int(i) == len(self.blocks):
                    nodes.append('EXIT')
                else:
                    st = self.blocks[i].stmts_to_code()
                    st_no_space = re.sub(r"\s+", "", st)
                    st_no_space = st_no_space.replace('"', "'")
                    # if start with if or while, delete these keywords
                    if st.startswith('if'):
                        st = st[3:]
                    elif st.startswith('while'):
                        st = st[6:]
                    if self.blocks[i].condition:
                        st = 'T '+ st
                    nodes.append(st)
                self.revert[len(nodes)] = i
        
    def track_execution(self):
        nodes = []
        blocks = []
        matching = {}
    
        # add
        code = {}
        for_loop = {}
        for i in self.blocks:
            if self.blocks[i].for_loop != 0:
                if self.blocks[i].for_loop not in for_loop:
                    for_loop[self.blocks[i].for_loop] = [i]
                else:
                    for_loop[self.blocks[i].for_loop].append(i)
        first = []
        second = []
        for i in for_loop:
            first.append(for_loop[i][0]+1)
            second.append(for_loop[i][1])
        orin_node = []
        track = {}
        track_for = {}
        for i in self.blocks:
            if self.blocks[i].stmts_to_code():
                if int(i) == 1:
                    st = 'BEGIN'
                elif int(i) == len(self.blocks):
                    st = 'EXIT'
                else:
                    if i in first:
                        line = astor.to_source(self.blocks[i].for_name)
                        st = line.split('\n')[0]
                        st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
                    else:
                        st = self.blocks[i].stmts_to_code()
                        st = re.sub(r"\s+", "", st).replace('"', "'").replace("(", "").replace(")", "")
                orin_node.append([i, st, None])
                if st not in track:
                    track[st] = [len(orin_node)-1]
                else:
                    track[st].append(len(orin_node)-1)
                track_for[i] = len(orin_node)-1
        with open(self.filename, 'r') as file:
            lines = file.readlines()
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
        orin_node[0][2] = 0
        orin_node[-1][2] = len(lines)+1
        for i in second:
            max_val = 0
            for edge in self.edges:
                if edge[0] == i:
                    if orin_node[track_for[edge[1]]][2] > max_val:
                        max_val = orin_node[track_for[edge[1]]][2]
                if edge[1] == i:
                    if orin_node[track_for[edge[0]]][2] > max_val:
                        max_val = orin_node[track_for[edge[0]]][2]
            orin_node[track_for[i]][2] = max_val + 0.5
        # sort orin_node by the third element
        orin_node.sort(key=lambda x: x[2])
        #add    
        for t in orin_node:
            i = t[0]
            if self.blocks[i].stmts_to_code():
                if int(i) == 1:
                    nodes.append('BEGIN')
                    blocks.append('BEGIN')
                elif int(i) == len(self.blocks):
                    nodes.append('EXIT')
                    blocks.append('EXIT')
                else:
                    st = self.blocks[i].stmts_to_code()
                    st_no_space = re.sub(r"\s+", "", st)
                    st_no_space = st_no_space.replace('"', "'")
                    blocks.append(st_no_space)
                    # if start with if or while, delete these keywords
                    if st.startswith('if'):
                        st = st[3:]
                    elif st.startswith('while'):
                        st = st[6:]
                    if self.blocks[i].condition:
                        st = 'T '+ st
                    nodes.append(st)
                matching[i] = len(nodes)
                self.revert[len(nodes)] = i

        edges = {}
        for edge in self.edges:
            if matching[edge[0]] not in edges:
                edges[matching[edge[0]]] = [matching[edge[1]]]
            else:
                edges[matching[edge[0]]].append(matching[edge[1]])

        for_loop = {}
        for i in self.blocks:
            if self.blocks[i].for_loop != 0:
                if self.blocks[i].for_loop not in for_loop:
                    for_loop[self.blocks[i].for_loop] = [matching[i]]
                else:
                    for_loop[self.blocks[i].for_loop].append(matching[i])
        store = {}
        last_loop = {}
        for i in for_loop:
            lst = for_loop[i]
            start = lst[0]
            end = lst[1]
            if self.blocks[self.revert[start]].condition:
                self.blocks[self.revert[start+1]].condition = True
            store[start+1] = [start]
            for t in matching:
                if matching[t] == start+1:
                    line = astor.to_source(self.blocks[t].for_name)
                    code = line.split('\n')[0]
                    st_no_space = re.sub(r"\s+", "", code)
                    st_no_space = st_no_space.replace('"', "'")
                    blocks[start] = st_no_space
            edges.pop(start)
            for j in edges:
                if start in edges[j]:
                    edges[j].remove(start)
                    edges[j].append(start + 1)
            for a in edges:
                if end in edges[a]:
                    edges[a].remove(end)
                    edges[a].append(start+1)
                    last_loop[a] = end
            edges.pop(end)
            if nodes[start+1].startswith('T'):
                store[start+1].append(start+2)
                for b in edges[start+2]:
                    edges[start+1].append(b)
                edges.pop(start+2)
                edges[start+1].remove(start+2)
            else:
                store[start+1].append(start+3)
                for b in edges[start+3]:
                    edges[start+1].append(b)
                edges.pop(start+3)
                edges[start+1].remove(start+3)
        t = trace_execution.Trace(ignoredirs=[sys.base_prefix, sys.base_exec_prefix,],
                            trace=0, count=1)
        arguments = []
        sys.argv = [self.filename, arguments]
        sys.path[0] = os.path.dirname(self.filename)

        with io.open_code(self.filename) as fp:
            code = compile(fp.read(), self.filename, 'exec')
        # try to emulate __main__ namespace as much as possible
        globs = {
            '__file__': self.filename,
            '__name__': '__main__',
            '__package__': None,
            '__cached__': None,
        }
        terminate = True
        try:
            t.runctx(code, globs, globs)
        except Exception as e:
            terminate = False

        source = linecache.getlines(self.filename)
        code_line = [element.lstrip().replace('\n', '') for element in source]
        execution_path = []
        for lineno in t.exe_path:
            no_spaces = re.sub(r"\s+", "", code_line[lineno-1])
            if no_spaces.startswith("if("):
                no_spaces = no_spaces.replace("(", "").replace(")", "")
            if no_spaces.startswith("elif"):
                no_spaces = no_spaces[2:]
            execution_path.append(no_spaces)
        
        check_True_condition = []
        for i in range(len(execution_path)-1):
            if execution_path[i].startswith('if') or execution_path[i].startswith('while') or execution_path[i].startswith('for'):
                if t.exe_path[i+1] == t.exe_path[i]+1:
                    check_True_condition.append(i)

        current_node = 1
        path = [self.revert[current_node]]
        exit_flag = False
        for s in range(len(execution_path)):
            node = execution_path[s]
            if current_node == 1:
                current_node = edges[current_node][0]
            else:
                c = 0
                if node == "break" or node == "continue":
                    continue
                if len(edges[current_node]) == 2:
                    if blocks[edges[current_node][0]-1] == blocks[edges[current_node][1]-1]:
                        if (s-1) in check_True_condition:
                            if self.blocks[self.revert[edges[current_node][0]]].condition:
                                current_node = edges[current_node][0]
                            else:
                                current_node = edges[current_node][1]
                        else:
                            if self.blocks[self.revert[edges[current_node][0]]].condition:
                                current_node = edges[current_node][1]
                            else:
                                current_node = edges[current_node][0]
                        if current_node in last_loop:
                            if terminate:
                                if self.revert[last_loop[current_node]] not in path:
                                    path.append(self.revert[last_loop[current_node]])
                            else:
                                if s != len(execution_path) - 1:
                                    if self.revert[last_loop[current_node]] not in path:
                                        path.append(self.revert[last_loop[current_node]])
                        if current_node in store:
                            for i in store[current_node]:
                                if self.revert[i] not in path:
                                    path.append(self.revert[i])
                        if self.revert[current_node] not in path:
                            path.append(self.revert[current_node])
                        continue
                for next_node in edges[current_node]:
                    c += 1
                    node = node.replace('"', "'")
                    # print(node)
                    # print(blocks[next_node-1])
                    # print("___________")
                    if blocks[next_node-1] == node:
                        if next_node in last_loop:
                            if terminate:
                                if self.revert[last_loop[next_node]] not in path:
                                    path.append(self.revert[last_loop[next_node]])
                            else:
                                if s != len(execution_path) - 1:
                                    if self.revert[last_loop[next_node]] not in path:
                                        path.append(self.revert[last_loop[next_node]])
                        if next_node in store:
                            for i in store[next_node]:
                                if self.revert[i] not in path:
                                    path.append(self.revert[i])
                        current_node = next_node
                        break
                    if c == len(edges[current_node]):
                        exit_flag = True
                        raise Exception(f"Error: Cannot find the execution path in CFG in file {self.filename}")
        
                if exit_flag:
                    break
            if self.revert[current_node] not in path:
                path.append(self.revert[current_node])
        if terminate:
            node_max = 0
            for i in self.blocks:
                if i > node_max:
                    node_max = i 
            path.append(node_max)

        self.path = path
    
    def clean(self):
        des_edges = {}
        for edge in self.edges:
            if edge[0] not in des_edges:
                des_edges[edge[0]] = [edge[1]]
            else:
                des_edges[edge[0]].append(edge[1])
        blank_node = []
        for node in des_edges:
            check = False
            if self.blocks[node].stmts_to_code():
                current = node
                for next in des_edges[node]:
                    next_node = next
                    while self.blocks[next_node].stmts_to_code() == '':
                        if next_node not in blank_node:
                            blank_node.append(next_node)
                        current = next_node
                        if current not in des_edges:
                            break
                        next_node = des_edges[current][0]
                        if (current, next_node) in self.back_edges:
                            check = True
                    if (node, next_node) not in self.edges:
                        self.edges[(node, next_node)] = None
                        if check:
                            self.back_edges.append((node, next_node))
                        if next_node != node and next_node not in self.blocks[node].next:
                            self.blocks[node].next.append(next_node)
            else:
                if node not in blank_node:
                    blank_node.append(node)
        for edge in self.edges.copy():
            if edge[0] in blank_node or edge[1] in blank_node:
                self.edges.pop(edge)
        for i in self.blocks:
            for node in blank_node:
                if node in self.blocks[i].next:
                    self.blocks[i].next.remove(node)
                
    def _show(self, fmt: str = 'pdf', calls: bool = True) -> gv.dot.Digraph:
        self.graph = gv.Digraph(name='cluster_' + self.name, format=fmt, graph_attr={'label': self.name})
        self._traverse(self.start, calls=calls)
        for k, v in self.func_calls.items():
            self.graph.subgraph(v._show(fmt, calls))
        return self.graph


    def show(self, filepath: str = './output', fmt: str = 'pdf', calls: bool = True, show: bool = True) -> None:
        self._show(fmt, calls)
        self.graph.render(filepath, view=show, cleanup=True)


class CFGVisitor(ast.NodeVisitor):

    invertComparators: Dict[Type[ast.AST], Type[ast.AST]] = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
                                                               ast.LtE: ast.Gt,
                                                               ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Is: ast.IsNot,
                                                               ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}

    def __init__(self):
        super().__init__()
        self.count_for_loop = 0
        self.loop_stack: List[BasicBlock] = []
        self.for_stack: List[int] = []
        self.continue_stack: List[BasicBlock] = []
        self.ifExp = False
        self.store_True: List[int] = []

    def build(self, name: str, tree: Type[ast.AST]) -> CFG:
        self.cfg = CFG(name)
        self.filename = name
        begin_block = self.new_block()
        begin_block.stmts = [ast.Expr(value=ast.Str(s='BEGIN'))]
        self.cfg.start = begin_block
        self.curr_block = self.new_block()
        self.add_edge(begin_block.bid, self.curr_block.bid)

        self.visit(tree)
        self.add_EXIT_node()
        self.remove_empty_blocks(self.cfg.start)
        for edge in self.cfg.lst_temp:
            if edge in self.cfg.back_edges:
                self.cfg.back_edges.remove(edge)
        return self.cfg

    def add_EXIT_node(self) -> None:
        exit_block = self.new_block()
        exit_block.stmts = [ast.Expr(value=ast.Str(s='EXIT'))]
        self.add_edge(self.curr_block.bid, exit_block.bid)
        
    def new_block(self) -> BasicBlock:
        bid: int = BlockId().gen()
        self.cfg.blocks[bid] = BasicBlock(bid)
        if bid in self.store_True:
            self.cfg.blocks[bid].condition = True
            self.store_True.remove(bid)
        return self.cfg.blocks[bid]

    def add_stmt(self, block: BasicBlock, stmt: Type[ast.AST]) -> None:
        # if block.stmts contain 1 stmt, then create the new node and add the stmt to the new node and add edge from block to new node
        if len(block.stmts) == 1:
            new_block = self.new_block()
            new_block.stmts.append(stmt)
            self.add_edge(block.bid, new_block.bid)
            self.curr_block = new_block
        else:
            block.stmts.append(stmt)

    def add_edge(self, frm_id: int, to_id: int, condition=None) -> BasicBlock:
        self.cfg.blocks[frm_id].next.append(to_id)
        self.cfg.blocks[to_id].prev.append(frm_id)       
        self.cfg.edges[(frm_id, to_id)] = condition
        return self.cfg.blocks[to_id]

    def add_loop_block(self) -> BasicBlock:
        if self.curr_block.is_empty() and not self.curr_block.has_next():
            return self.curr_block
        else:
            loop_block = self.new_block()
            self.add_edge(self.curr_block.bid, loop_block.bid)
            return loop_block

    def add_subgraph(self, tree: Type[ast.AST]) -> None:
        self.cfg.func_calls[tree.name] = CFGVisitor().build(tree.name, ast.Module(body=tree.body))

    def add_condition(self, cond1: Optional[Type[ast.AST]], cond2: Optional[Type[ast.AST]]) -> Optional[Type[ast.AST]]:
        if cond1 and cond2:
            return ast.BoolOp(ast.And(), values=[cond1, cond2])
        else:
            return cond1 if cond1 else cond2

    # not tested
    def remove_empty_blocks(self, block: BasicBlock, visited: Set[int] = set()) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            if block.is_empty():
                for prev_bid in block.prev:
                    prev_block = self.cfg.blocks[prev_bid]
                    for next_bid in block.next:
                        next_block = self.cfg.blocks[next_bid]
                        self.add_edge(prev_bid, next_bid, self.add_condition(self.cfg.edges.get((prev_bid, block.bid)), self.cfg.edges.get((block.bid, next_bid))))
                        self.cfg.edges.pop((block.bid, next_bid), None)
                        if (block.bid, next_bid) in self.cfg.back_edges:
                            self.cfg.back_edges.append((prev_bid, next_bid))
                            self.cfg.lst_temp.append((block.bid, next_bid))
                        next_block.remove_from_prev(block.bid)
                    self.cfg.edges.pop((prev_bid, block.bid))
                    if (prev_bid, block.bid) in self.cfg.back_edges:
                        self.cfg.back_edges.append((prev_bid, next_bid))
                        self.cfg.lst_temp.append((prev_bid, block.bid))
                    prev_block.remove_from_next(block.bid)
                block.prev.clear()
                for next_bid in block.next:
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)
                block.next.clear()

            else:
                for next_bid in block.next:
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)

    def invert(self, node: Type[ast.AST]) -> Type[ast.AST]:
        if type(node) == ast.Compare:
            if len(node.ops) == 1:
                return ast.Compare(left=node.left, ops=[self.invertComparators[type(node.ops[0])]()], comparators=node.comparators)
            else:
                tmpNode = ast.BoolOp(op=ast.And(), values = [ast.Compare(left=node.left, ops=[node.ops[0]], comparators=[node.comparators[0]])])
                for i in range(0, len(node.ops) - 1):
                    tmpNode.values.append(ast.Compare(left=node.comparators[i], ops=[node.ops[i+1]], comparators=[node.comparators[i+1]]))
                return self.invert(tmpNode)
        elif isinstance(node, ast.BinOp) and type(node.op) in self.invertComparators:
            return ast.BinOp(node.left, self.invertComparators[type(node.op)](), node.right)
        elif type(node) == ast.NameConstant and type(node.value) == bool:
            return ast.NameConstant(value=not node.value)
        elif type(node) == ast.BoolOp:
            return ast.BoolOp(values = [self.invert(x) for x in node.values], op = {ast.And: ast.Or(), ast.Or: ast.And()}.get(type(node.op)))
        elif type(node) == ast.UnaryOp:
            return self.UnaryopInvert(node)
        else:
            return ast.UnaryOp(op=ast.Not(), operand=node)

    def UnaryopInvert(self, node: Type[ast.AST]) -> Type[ast.AST]:
        if type(node.op) == ast.UAdd:
            return ast.UnaryOp(op=ast.USub(),operand = node.operand)
        elif type(node.op) == ast.USub:
            return ast.UnaryOp(op=ast.UAdd(),operand = node.operand)
        elif type(node.op) == ast.Invert:
            return ast.UnaryOp(op=ast.Not(), operand=node)
        else:
            return node.operand

    # def boolinvert(self, node:Type[ast.AST]) -> Type[ast.AST]:
    #     value = []
    #     for item in node.values:
    #         value.append(self.invert(item))
    #     if type(node.op) == ast.Or:
    #         return ast.BoolOp(values = value, op = ast.And())
    #     elif type(node.op) == ast.And:
    #         return ast.BoolOp(values = value, op = ast.Or())

    def combine_conditions(self, node_list: List[Type[ast.AST]]) -> Type[ast.AST]:
        return node_list[0] if len(node_list) == 1 else ast.BoolOp(op=ast.And(), values = node_list)

    def generic_visit(self, node):
        if type(node) in [ast.Import, ast.ImportFrom]:
            self.add_stmt(self.curr_block, node)
            return
        if type(node) in [ast.FunctionDef, ast.AsyncFunctionDef]:
            self.add_stmt(self.curr_block, node)
            self.add_subgraph(node)
            return
        if type(node) in [ast.AnnAssign, ast.AugAssign]:
            self.add_stmt(self.curr_block, node)
        super().generic_visit(node)

    def get_function_name(self, node: Type[ast.AST]) -> str:
        if type(node) == ast.Name:
            return node.id
        elif type(node) == ast.Attribute:
            return self.get_function_name(node.value) + '.' + node.attr
        elif type(node) == ast.Str:
            return node.s
        elif type(node) == ast.Subscript:
            return node.value.id
        elif type(node) == ast.Lambda:
            return 'lambda function'

    def populate_body(self, body_list: List[Type[ast.AST]], to_bid: int, type: str) -> None:
        for child in body_list:
            self.visit(child)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, to_bid)
            if type == "While" or type == "Try":
                self.cfg.back_edges.append((self.curr_block.bid, to_bid))
    
    def populate_If_body(self, body_list: List[Type[ast.AST]], node_list: List[int]):
        for child in body_list:
            self.visit(child)
        if not self.curr_block.next:
            node_list.append(self.curr_block.bid)
        return node_list
        
    def populate_For_body(self, body_list: List[Type[ast.AST]]) -> None:
        for child in body_list:
            self.visit(child)
        new_node = self.new_block()
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, new_node.bid)
        return new_node

    # assert type check
    def visit_Assert(self, node):
        self.add_stmt(self.curr_block, node)
        # If the assertion fails, the current flow ends, so the fail block is a
        # final block of the CFG.
        # self.cfg.finalblocks.append(self.add_edge(self.curr_block.bid, self.new_block().bid, self.invert(node.test)))
        # If the assertion is True, continue the flow of the program.
        # success block
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)
        self.generic_visit(node)

    # TODO: change all those registers to stacks!
    def visit_Assign(self, node):
        if type(node.value) in [ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda] and len(node.targets) == 1 and type(node.targets[0]) == ast.Name: # is this entire statement necessary?
            if type(node.value) == ast.ListComp:
                self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
                self.listCompReg = (node.targets[0].id, node.value)
            elif type(node.value) == ast.SetComp:
                self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Call(func=ast.Name(id='set', ctx=ast.Load()), args=[], keywords=[])))
                self.setCompReg = (node.targets[0].id, node.value)
            elif type(node.value) == ast.DictComp:
                self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Dict(keys=[], values=[])))
                self.dictCompReg = (node.targets[0].id, node.value)
            elif type(node.value) == ast.GeneratorExp:
                self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())], value=ast.Call(func=ast.Name(id='__' + node.targets[0].id + 'Generator__', ctx=ast.Load()), args=[], keywords=[])))
                self.genExpReg = (node.targets[0].id, node.value)
            else:
                self.lambdaReg = (node.targets[0].id, node.value)
        else:
            self.add_stmt(self.curr_block, node)
        self.generic_visit(node)

    def visit_Await(self, node):
        afterawait_block = self.new_block()
        self.add_edge(self.curr_block.bid, afterawait_block.bid)
        self.generic_visit(node)
        self.curr_block = afterawait_block

    def visit_Break(self, node):
        assert len(self.loop_stack), "Found break not inside loop"
        self.add_edge(self.curr_block.bid, self.loop_stack[-1].bid, ast.Break())

    def visit_Continue(self, node):
        assert len(self.continue_stack), "Found continue not inside loop"
        self.add_edge(self.curr_block.bid, self.continue_stack[-1].bid, ast.Continue())
    
    def visit_Call(self, node):
        if type(node.func) == ast.Lambda:
            self.lambdaReg = ('Anonymous Function', node.func)
            self.generic_visit(node)
        # else:
        #     self.curr_block.calls.append(self.get_function_name(node.func))

    def visit_DictComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            if self.dictCompReg[0]: # bug if there is else statement in comprehension
                return [ast.Assign(targets=[ast.Subscript(value=ast.Name(id=self.dictCompReg[0], ctx=ast.Load()), slice=ast.Index(value=self.dictCompReg[1].key), ctx=ast.Store())], value=self.dictCompReg[1].value)]
            # else: # not supported yet
            #     return [ast.Expr(value=self.dictCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_DictComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_DictComp_Rec(generators[:-1]), orelse=[])]

    def visit_DictComp(self, node):
        try: # try may change to checking if self.dictCompReg exists
            self.generic_visit(ast.Module(self.visit_DictComp_Rec(self.dictCompReg[1].generators)))
        except:
            pass
        finally:
            self.dictCompReg = None

    # ignore the case when using set or dict comprehension or generator expression but the result is not assigned to a variable
    def visit_Expr(self, node):
        if type(node.value) == ast.ListComp and type(node.value.elt) == ast.Call:
            self.listCompReg = (None, node.value)
        elif type(node.value) == ast.Lambda:
            self.lambdaReg = ('Anonymous Function', node.value)
        # elif type(node.value) == ast.Call and type(node.value.func) == ast.Lambda:
        #     self.lambdaReg = ('Anonymous Function', node.value.func)
        else:
            self.add_stmt(self.curr_block, node)
        self.generic_visit(node)

    def visit_For(self, node):
        self.count_for_loop += 1
        assign_block = self.new_block()
        if self.curr_block.condition:
            assign_block.condition = True
        self.add_edge(self.curr_block.bid, assign_block.bid)
        self.curr_block = assign_block
        self.curr_block.for_loop = self.count_for_loop
        self.for_stack.append(self.count_for_loop)
        num = self.count_for_loop
        self.add_stmt(self.curr_block, ast.Assign(targets=[ast.Name(id=f'p{num}', ctx=ast.Store())], value=ast.Num(n=0)))
        if_node = ast.If(test=ast.Compare(left=ast.Name(id=f'p{num}', ctx=ast.Load()), ops=[ast.Lt()], comparators=[ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[node.iter], keywords=[])]), body=[ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=ast.Subscript(value=node.iter, slice=ast.Index(value=ast.Name(id=f'p{num}', ctx=ast.Load())), ctx=ast.Load()))] + node.body, orelse=[])    
        loop_guard = self.add_loop_block()
        self.continue_stack.append(loop_guard)
        self.curr_block = loop_guard
        self.curr_block.for_name = node
        self.add_stmt(self.curr_block, if_node)
        # New block for the body of the for-loop.
        for_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        if not node.orelse:
            # Block of code after the for loop.
            afterfor_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            self.loop_stack.append(afterfor_block)
            self.curr_block = for_block
            self.curr_block.condition = True
            self.curr_block = self.populate_For_body(if_node.body)
            end_node = ast.AugAssign(target=ast.Name(id=f'p{num}', ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1))
            self.add_stmt(self.curr_block, end_node)
            self.curr_block.for_loop = self.for_stack[-1]
            self.add_edge(self.curr_block.bid, loop_guard.bid)
            self.cfg.back_edges.append((self.curr_block.bid, loop_guard.bid))
        else:
            # Block of code after the for loop.
            afterfor_block = self.new_block()
            orelse_block = self.add_edge(self.curr_block.bid, self.new_block().bid, ast.Name(id='else', ctx=ast.Load()))
            self.loop_stack.append(afterfor_block)
            self.curr_block = for_block
            self.curr_block.condition = True
            self.curr_block = self.populate_For_body(if_node.body)
            end_node = ast.AugAssign(target=ast.Name(id=f'p{num}', ctx=ast.Store()), op=ast.Add(), value=ast.Num(n=1))
            self.add_stmt(self.curr_block, end_node)
            self.curr_block.for_loop = self.for_stack[-1]
            self.add_edge(self.curr_block.bid, loop_guard.bid)
            self.cfg.back_edges.append((self.curr_block.bid, loop_guard.bid))

            self.curr_block = orelse_block
            for child in node.orelse:
                self.visit(child)
            self.add_edge(orelse_block.bid, afterfor_block.bid, "For")
        
        # Continue building the CFG in the after-for block.
        self.curr_block = afterfor_block     
        self.continue_stack.pop()  
        self.loop_stack.pop()   
        self.for_stack.pop()

    def visit_GeneratorExp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.genExpReg[1].elt) # the location of the node may be wrong
            if self.genExpReg[0]: # bug if there is else statement in comprehension
                return [ast.Expr(value=ast.Yield(value=self.genExpReg[1].elt))]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_GeneratorExp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_GeneratorExp_Rec(generators[:-1]), orelse=[])]

    def visit_GeneratorExp(self, node):
        try: # try may change to checking if self.genExpReg exists
            self.generic_visit(ast.FunctionDef(name='__' + self.genExpReg[0] + 'Generator__', 
                args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body = self.visit_GeneratorExp_Rec(self.genExpReg[1].generators), 
                decorator_list=[], returns=None))
        except:
            pass
        finally:
            self.genExpReg = None

    def visit_If(self, node):
        # Add the If statement at the end of the current block.
        self.add_stmt(self.curr_block, node)

        # Create a block for the code after the if-else.
        body_block = self.new_block()
        self.store_True.append(body_block.bid+1)
        if_block = self.add_edge(self.curr_block.bid, body_block.bid, node.test)
        node_list = []
        # New block for the body of the else if there is an else clause.
        store = self.curr_block.bid
        node_list = self.populate_If_body(node.body, node_list)
        if node.orelse:
            self.curr_block = self.add_edge(store, self.new_block().bid, self.invert(node.test))

            # Visit the children in the body of the else to populate the block.
            node_list = self.populate_If_body(node.orelse, node_list)
        else:
            node_list.append(store)
        # Visit children to populate the if block.
        self.curr_block = if_block
        afterif_block = self.new_block()
        for bid in node_list:
            self.add_edge(bid, afterif_block.bid)
        # Continue building the CFG in the after-if block.
        self.curr_block = afterif_block

    def visit_IfExp_Rec(self, node: Type[ast.AST]) -> List[Type[ast.AST]]:
        return [ast.If(test=node.test, body=[ast.Return(value=node.body)], orelse=self.visit_IfExp_Rec(node.orelse) if type(node.orelse) == ast.IfExp else [ast.Return(value=node.orelse)])]

    def visit_IfExp(self, node):
        if self.ifExp:
            self.generic_visit(ast.Module(self.visit_IfExp_Rec(node)))

    def visit_Lambda(self, node): # deprecated since there is autopep8
        self.add_subgraph(ast.FunctionDef(name=self.lambdaReg[0], args=node.args, body = [ast.Return(value=node.body)], decorator_list=[], returns=None))
        self.lambdaReg = None

    def visit_ListComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.listCompReg[1].elt) # the location of the node may be wrong
            if self.listCompReg[0]: # bug if there is else statement in comprehension
                return [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.listCompReg[0], ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[self.listCompReg[1].elt], keywords=[]))]
            else:
                return [ast.Expr(value=self.listCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_ListComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_ListComp_Rec(generators[:-1]), orelse=[])]

    def visit_ListComp(self, node):
        try: # try may change to checking if self.listCompReg exists
            self.generic_visit(ast.Module(self.visit_ListComp_Rec(self.listCompReg[1].generators)))
        except:
            pass
        finally:
            self.listCompReg = None

    def visit_Pass(self, node):
        self.add_stmt(self.curr_block, node)

    def visit_Raise(self, node):
        self.add_stmt(self.curr_block, node)
        self.curr_block = self.new_block()

    # ToDO: final blocks to be add
    def visit_Return(self, node):
        if type(node.value) == ast.IfExp:
            self.ifExp = True
            self.generic_visit(node)
            self.ifExp = False
        else:
            self.add_stmt(self.curr_block, node)
        # self.cfg.finalblocks.append(self.curr_block)
        # Continue in a new block but without any jump to it -> all code after
        # the return statement will not be included in the CFG.
        self.curr_block = self.new_block()

    def visit_SetComp_Rec(self, generators: List[Type[ast.AST]]) -> List[Type[ast.AST]]:
        if not generators:
            self.generic_visit(self.setCompReg[1].elt) # the location of the node may be wrong
            if self.setCompReg[0]:
                return [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=self.setCompReg[0], ctx=ast.Load()), attr='add', ctx=ast.Load()), args=[self.setCompReg[1].elt], keywords=[]))]
            else: # not supported yet
                return [ast.Expr(value=self.setCompReg[1].elt)]
        else:
            return [ast.For(target=generators[-1].target, iter=generators[-1].iter, body=[ast.If(test=self.combine_conditions(generators[-1].ifs), body=self.visit_SetComp_Rec(generators[:-1]), orelse=[])] if generators[-1].ifs else self.visit_SetComp_Rec(generators[:-1]), orelse=[])]

    def visit_SetComp(self, node):
        try: # try may change to checking if self.setCompReg exists
            self.generic_visit(ast.Module(self.visit_SetComp_Rec(self.setCompReg[1].generators)))
        except:
            pass
        finally:
            self.setCompReg = None

    def visit_Try(self, node):
        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        self.add_stmt(loop_guard, ast.Try(body=[], handlers=[], orelse=[], finalbody=[]))

        after_try_block = self.new_block()
        self.add_stmt(after_try_block, ast.Name(id='handle errors', ctx=ast.Load()))
        self.populate_body(node.body, after_try_block.bid, "Try")

        self.curr_block = after_try_block

        if node.handlers:
            for handler in node.handlers:
                before_handler_block = self.new_block()
                self.curr_block = before_handler_block
                self.add_edge(after_try_block.bid, before_handler_block.bid, handler.type if handler.type else ast.Name(id='Error', ctx=ast.Load()))

                after_handler_block = self.new_block()
                self.add_stmt(after_handler_block, ast.Name(id='end except', ctx=ast.Load()))
                self.populate_body(handler.body, after_handler_block.bid, "Try")
                self.add_edge(after_handler_block.bid, after_try_block.bid)

        if node.orelse:
            before_else_block = self.new_block()
            self.curr_block = before_else_block
            self.add_edge(after_try_block.bid, before_else_block.bid, ast.Name(id='No Error', ctx=ast.Load()))

            after_else_block = self.new_block()
            self.add_stmt(after_else_block, ast.Name(id='end no error', ctx=ast.Load()))
            self.populate_body(node.orelse, after_else_block.bid, "Try")
            self.add_edge(after_else_block.bid, after_try_block.bid)

        finally_block = self.new_block()
        self.curr_block = finally_block

        if node.finalbody:
            self.add_edge(after_try_block.bid, finally_block.bid, ast.Name(id='Finally', ctx=ast.Load()))
            after_finally_block = self.new_block()
            self.populate_body(node.finalbody, after_finally_block.bid, "Try")
            self.curr_block = after_finally_block
        else:
            self.add_edge(after_try_block.bid, finally_block.bid)

    def visit_While(self, node):
        loop_guard = self.add_loop_block()
        self.continue_stack.append(loop_guard)
        self.curr_block = loop_guard
        self.add_stmt(loop_guard, node)
        # New block for the case where the test in the while is False.
        afterwhile_block = self.new_block()
        self.loop_stack.append(afterwhile_block)
        inverted_test = self.invert(node.test)

        if not node.orelse:
            # Skip shortcut loop edge if while True:
            if not (isinstance(inverted_test, ast.NameConstant) and inverted_test.value == False):
                self.add_edge(self.curr_block.bid, afterwhile_block.bid, inverted_test)

            # New block for the case where the test in the while is True.
            # Populate the while block.
            body_block = self.new_block()
            body_block.condition = True
            self.curr_block = self.add_edge(self.curr_block.bid, body_block.bid, node.test)
            self.populate_body(node.body, loop_guard.bid, "While")
        else:
            orelse_block = self.new_block()
            if not (isinstance(inverted_test, ast.NameConstant) and inverted_test.value == False):
                self.add_edge(self.curr_block.bid, orelse_block.bid, inverted_test)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid, node.test)

            self.populate_body(node.body, loop_guard.bid, "While")
            self.curr_block = orelse_block
            for child in node.orelse:
                self.visit(child)
            self.add_edge(orelse_block.bid, afterwhile_block.bid)
            
        # Continue building the CFG in the after-while block.
        self.curr_block = afterwhile_block
        self.loop_stack.pop()
        self.continue_stack.pop()

    def visit_Yield(self, node):
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)


class PyParser:

    def __init__(self, script):
        self.script = script

    def formatCode(self):
        self.script = autopep8.fix_code(self.script)

    # https://github.com/liftoff/pyminifier/blob/master/pyminifier/minification.py
    def removeCommentsAndDocstrings(self):
        io_obj = io.StringIO(self.script)  # ByteIO for Python2?
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        # Note regarding NEWLINE vs NL: The tokenize module
                        # differentiates between newlines that start a new statement
                        # and newlines inside of operators such as parens, brackes,
                        # and curly braces.  Newlines inside of operators are
                        # NEWLINE and newlines that start new code are NL.
                        # Catch whole-module docstrings:
                        if start_col > 0:
                            # Unlabelled indentation means we're inside an operator
                            out += token_string
                        # Note regarding the INDENT token: The tokenize module does
                        # not label indentation inside of an operator (parens,
                        # brackets, and curly braces) as actual indentation.
                        # For example:
                        # def foo():
                        #     "The spaces before this docstring are tokenize.INDENT"
                        #     test = [
                        #         "The spaces before this string do not get a token"
                        #     ]
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        self.script = out

if __name__ == '__main__':
    filename = sys.argv[1]
    file = filename.split('/')[-1].split('.')[0]
    filepath = f'./CFG/{file}_CFG'
    try:
        source = open(filename, 'r').read()
        compile(source, filename, 'exec')
    except:
        print('Error in source code')
        exit(1)

    parser = PyParser(source)
    parser.removeCommentsAndDocstrings()
    parser.formatCode()
    cfg = CFGVisitor().build(filename, ast.parse(parser.script))
    cfg.clean()
    cfg.track_execution() 
    cfg.show(filepath)
    
    