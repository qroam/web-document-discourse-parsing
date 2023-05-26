import os
from tqdm import tqdm
from typing import List

from visualization.reader import WebPage, LabelReader


open_hyper_marks = ["<h1>",
                    "<h2>&nbsp",
                    "<h3>&nbsp&nbsp",
                    "<h4>&nbsp&nbsp&nbsp",
                    "<h5>&nbsp&nbsp&nbsp&nbsp",
                    "<h6>&nbsp&nbsp&nbsp&nbsp&nbsp",
                    "<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp",
                    "<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp",
                    "<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp",
                    "<p>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp"]
close_hyper_marks = ["</h1>","</h2>","</h3>","</h4>","</h5>","</h6>","</p>","</p>","</p>","</p>"]
# nobr_open = '<nobr style="color:#FF0000";>'
nobr_open = lambda color: f'<nobr style="color:{color}";>'
colors = ["#FF0000","#FF7400","#CBD100","#6BB500","#01FF00","#00FFCC","#00F2FF","#009CFF","#0019FF","#7900FF"]
nobr_close = "</nobr>"
HTM_PREFIX = '<!DOCTYPE html><html><head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><title>大标题</title></head><body>'
HTM_SUFFIX = '</body></html>'

def node_to_html(node_id, node_text, node_depth):
    if node_id == -1:
        return '<p><nobr style="color:#D100D0";>======A New Sub-Discourse======</nobr></p>'
    return open_hyper_marks[node_depth] + nobr_open(colors[node_depth]) + str(node_id) + nobr_close + node_text + close_hyper_marks[node_depth]


"""def dfp(father_id_list, text_list, node_id=-1, depth=-1, previous_list=None, previous_id=-1):
    # current_node = -1
    # print(node_id)
    # print(father_id_list)
    html_string = ""
    structure_dict = {}
    node_depth = {}

    children_list = find_children(father_id_list, node_id, previous_list)
    structure_dict[node_id] = children_list
    node_depth[node_id] = depth
    if node_id != -1:
        html_string += open_hyper_marks[depth] + nobr_open(colors[depth]) + str(node_id) + nobr_close + text_list[node_id] + close_hyper_marks[depth]
    else:  # TODO
        if children_list == []:
            return structure_dict, node_depth, html_string
        html_string += '<p><nobr style="color:#D100D0";>======A New Sub-Discourse======</nobr></p>'

    for i, child_id in enumerate(children_list):
        # if previous_list[child_id] == 1 and node_id != -1 and i != 0:  # Break, repeat father node
        if previous_list[child_id] == 1 and i != 0:  # Break, repeat father node  # TODO
            # i != 0 to neglect the first child node, otherwise will trigger an infinite recursion
            # structure_dict_of_self, node_depth_of_self, html_string_of_self = dfp(father_id_list, text_list, node_id, depth, previous_list)  # 这会造成递归溢出，因为递归变量完全没有改变
            # html_string += open_hyper_marks[depth] + nobr_open(colors[depth]) + str(node_id) + nobr_close + text_list[node_id] + close_hyper_marks[depth]
            # structure_dict_of_child_node, node_depth_of_child_node, html_string_of_child_node = dfp(father_id_list, text_list, node_id=-1, depth=-1, previous_list=previous_list, previous_id=-1)
            structure_dict_of_child_node, node_depth_of_child_node, html_string_of_child_node = dfp(father_id_list,
                                                                                                    text_list,
                                                                                                    node_id=-1,
                                                                                                    depth=-1,
                                                                                                    previous_list=previous_list,
                                                                                                    previous_id=-1)

            structure_dict.update(structure_dict_of_child_node)
            node_depth.update(node_depth_of_child_node)
            html_string += html_string_of_child_node
            father_id_list[children_list[i-1]] = -9999  # TODO
            return structure_dict, node_depth, html_string  # TODO
        else:
            structure_dict_of_child_node, node_depth_of_child_node, html_string_of_child_node = dfp(father_id_list, text_list, child_id, depth+1, previous_list)
            structure_dict.update(structure_dict_of_child_node)
            node_depth.update(node_depth_of_child_node)
        # if previous_list[child_id] == 2:  # Combine
        #     assert father_id_list[child_id] == father_id_list[child_id-1], str(father_id_list[child_id]) + " " +str(father_id_list[child_id-1])
        #     html_string = html_string[:-len(close_hyper_marks[depth+1])] + nobr_open(colors[depth+1]) + str(child_id) + nobr_close + text_list[child_id] + close_hyper_marks[depth+1]
        # else:
        #     html_string += html_string_of_child_node
            html_string += html_string_of_child_node
        # 在子递归中是无法对父递归中的变量进行修改的。只能够修改全局变量，或者在父递归中通过关于自递归全局变量的判断，来在父递归中对父递归的变量进行修改
        # father_id_list[child_id] = -9999  # Search Complete
    return structure_dict, node_depth, html_string"""


def find_children(father_id_list, node_id, previous_list=None):
    """
    Find all the childen nodes of the input node `node_id`, given the `father_id_list` of the whole document
    :params
    father_id_list: List[int], father node id of each paragraph in the document
    node_id: int, the node you want to find its children nodes
    """
    children_list = []
    for node, father in enumerate(father_id_list):
        if father == node_id:
            children_list.append(node)
    return children_list


def dfp(father_id_list, node_id=-1, depth=-1):
    """
    Depth First Search
    """
    structure_dict = {}  # to record the subtree rooted by node_id, in the format node_id : List[child_nodes]
    node_depth = {}  # to record the absolute depth (in the whole tree) of nodes in the subtree rooted by node_id
    children_list = find_children(father_id_list, node_id)
    structure_dict[node_id] = children_list
    node_depth[node_id] = depth
    for child_node in children_list:
        structure_dict_of_child, node_depth_of_child = dfp(father_id_list, child_node, depth+1)
        structure_dict.update(structure_dict_of_child)
        node_depth.update(node_depth_of_child)
    return structure_dict, node_depth


def get_depth(node_depth, node_list):
    node_depth[-1] = -1
    if type(node_list) == int:
        return node_depth[node_list]
    return [node_depth[i] for i in node_list]

def get_continue_block(node_id, structure_dict, previous_relation_list):
    children_nodes = structure_dict[node_id]
    if children_nodes == []:
        return []
    previous_relations = [previous_relation_list[i] for i in children_nodes][1:]
    continue_block = []
    buffer = [children_nodes[0]]
    for i, relation in enumerate(previous_relations):
        if relation != 1:
            buffer.append(children_nodes[i + 1])
        else:
            continue_block.append(buffer)
            buffer = [children_nodes[i + 1]]
    continue_block.append(buffer)
    return continue_block

def check_satisfiability(children_continue_blocks_list):
    zero = 0
    one = 0
    multiple = 0
    for children_continue_block in children_continue_blocks_list:
        if len(children_continue_block) == 0:
            zero += 1
        elif len(children_continue_block) == 1:
            one += 1
        elif len(children_continue_block) > 1:
            multiple += 1
    if multiple > 1:
        return False
    elif multiple == 1:
        if one > 0:
            return False
        else:
            return True
    else:
        return True



def get_path(structure_dict, node_depth, previous_relation_list, current_path, current_depth=-1):
    paths = []
    # current_path = [[-1],]
    update_paths = []
    for path in current_path:
        depth = get_depth(node_depth, path)
        node_id_to_expand = [node_id for i, node_id in enumerate(path) if depth[i]==current_depth]
        children_continue_blocks = {}
        for i, node_id in enumerate(node_id_to_expand):
            continue_block_of_node: List[List[int]] = get_continue_block(node_id, structure_dict, previous_relation_list)
            children_continue_blocks[node_id] = continue_block_of_node
            # TODO: 对于每个互相连续的节点的children_continue_block的数量，只允许均为1(或0)，或者一个为多个、其余为0个. 否则会出现分配困难

        if len(node_id_to_expand) > 1:
            # assert check_satisfiability(list(children_continue_blocks.values())), children_continue_blocks.values()
            if not check_satisfiability(list(children_continue_blocks.values())):  # TODO
                print(f"Warning, some disapproving structure occures: {children_continue_blocks.values()}")
                universial_continue_blocks = [{father_id:node} for father_id, block in children_continue_blocks.items() for node in
                                              block]
            else:
                universial_continue_blocks = []
                for father_id, continue_blocks in children_continue_blocks.items():
                    if universial_continue_blocks == []:
                        universial_continue_blocks = [{father_id:block} for block in continue_blocks]
                    else:
                        for block_dict in universial_continue_blocks:
                            for new_block in continue_blocks:
                                block_dict.update({father_id:new_block})
        elif len(node_id_to_expand) == 1:
            universial_continue_blocks = [{list(children_continue_blocks.keys())[0]:block} for block in list(children_continue_blocks.values())[0]]
        else:
            universial_continue_blocks = []

        if universial_continue_blocks == []:  # TODO
            update_paths.append(path)
        else:
            for bifurcate in universial_continue_blocks:
                bifurcate_path = path
                for father_id, children_block in bifurcate.items():
                    bifurcate_path = bifurcate_path[:bifurcate_path.index(father_id)+1] + children_block + bifurcate_path[bifurcate_path.index(father_id)+1:]
                update_paths.append(bifurcate_path)

        """if len(node_id_to_expand) > 1:
            # assert check_satisfiability(list(children_continue_blocks.values())), children_continue_blocks.values()
            if not check_satisfiability(list(children_continue_blocks.values())):  # TODO
                universial_continue_blocks = [node for block in list(children_continue_blocks.values()) for node in block]
                print(f"Warning, some disapproving structure occures: {children_continue_blocks.values()}")
            else:
                universial_continue_blocks = []
                for continue_blocks in children_continue_blocks.values():
                    if universial_continue_blocks == []:
                        universial_continue_blocks = continue_blocks
                    else:
                        for block in universial_continue_blocks:
                            for new_block in continue_blocks:
                                block += new_block
        elif len(node_id_to_expand) == 1:
            universial_continue_blocks = list(children_continue_blocks.values())[0]
        else:
            universial_continue_blocks = []

        update_paths += [path + block for block in universial_continue_blocks]"""
        """if universial_continue_blocks == []:  # TODO
            update_paths.append(path)"""
    # print(update_paths)
    return update_paths

def get_path_total(tree_depth, structure_dict, node_depth, previous_relation_list):
    depth = -1
    current_path = [[-1], ]
    while(depth < tree_depth):
        update_paths = get_path(structure_dict, node_depth, previous_relation_list, current_path, current_depth=depth)
        current_path = update_paths
        depth += 1
    # return [sorted(i) for i in current_path]
    return current_path

def main(father_id_list, previous_relation_list):
    structure_dict, node_depth = dfp(father_id_list)
    tree_depth = max(list(node_depth.values()))
    # print(tree_depth)
    return get_path_total(tree_depth, structure_dict, node_depth, previous_relation_list)

def draw_html(total_path, node_depth, text_list):
    html_string = ""
    html_string += HTM_PREFIX
    for path in total_path:
        for node_id in path:
            html_string += node_to_html(node_id, text_list[node_id], node_depth[node_id])
    html_string += HTM_SUFFIX
    return html_string

"""class WebTreeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.node_depth = -1
        self.content = ""

        self.father_id = -1
        self.father = None
        self.previous_id = -1
        self.previous_node = -1
        self.previous_relation = 3  # 3 for NA, the first node of siblings

        self.children_id_list = []
        self.children_list = []

def build_webtree(structure_dict, node_depth, text_list, previous_relation_list):
    tree_node_list = [WebTreeNode(i) for i in range(len(text_list))]
    for i, content in enumerate(text_list):
        tree_node_list[i].content = content
        tree_node_list[i].children_id_list = structure_dict[i]
        if len(structure_dict[i]) > 1:
            for k in range(len(structure_dict[i]) - 1):
                tree_node_list[structure_dict[i][k+1]].previous_id = structure_dict[i][k]
                tree_node_list[structure_dict[i][k+1]].previous_node = tree_node_list[structure_dict[i][k]]
                tree_node_list[structure_dict[i][k+1]].previous_relation = previous_relation_list[structure_dict[i][k+1]]
        for j in structure_dict[i]:
            tree_node_list[i].children_list.append(tree_node_list[j])
            tree_node_list[j].father_id = i
            tree_node_list[j].father = tree_node_list[i]
        tree_node_list[i].node_depth = node_depth[i]

    root_node = WebTreeNode(-1)
    root_node.content = "======A New Sub-Discourse======"
    root_node.children_id_list = structure_dict[-1]
    root_node.children_list = [tree_node_list[j] for j in structure_dict[-1]]

    return root_node, tree_node_list


def combine_node(sibling_node_lists, previous_relation_list):
    pass

def combine(sibling_list, previous_relation_list):
    for i, node_id in enumerate(sibling_list):
        if previous_relation_list[node_id] == 0:  # Continue
            pass
        elif previous_relation_list[node_id] == 1:  # Break
            if i == 0:  # first node in siblings
                pass
            else:
                pass
        elif previous_relation_list[node_id] == 2:  # Combine
            pass

"""

# class StructureGenerator:
#     def __init__(self):
#         pass

if __name__ == '__main__':
    father_precdict = [-1, -1, 0, -1, -1, 0, 0, 0, -1, 0, 9, 0, 9, 9, 9, 0, 9, 0, 9, 9, 9, 9, 9, 9, 20, 0, 9, 9, 9, 28, 28, 23, 28, 28, 28, 28, 28, 28, 9, 9, 0, 0, 9, 9, 0, 9, 9, 0, -1, 0, 9, 50, 0, 0, 9, 0, 9, -1, 9, 9, 58, 58, 58, 58, 58, 58, 48, 58, 67, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 78, 78, 78, 58, 58, 58, 58, 58, 58, 0, 0, 0, 90, 0, 0, 0, 9, 0, 0, 9, 9, -1, 0, 0, 0, 9, 9, 105, 0, 0, 9, 0, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 120, 9, 122, 122, 122, 122, 122, 122, 122, 0, -1, -1, -1, 0, -1, 0, 0, 9, -1, 139, 139, 139, 142, 0, 144, 144]
    print(dfp(father_precdict))
    # exit(10)
    father909 = [-1, -1, -1, -1, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16, 0, 18, 0, 20, 0, 22, -1, -1, -1, -1, -1]
    previous909 = [3, 1, 2, 2, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1]
    print(main(father909, previous909))
    exit(10)

    reader = LabelReader()
    label_file_path = "robertawwmext.txt"
    source_file_dir = "../webpage_discourse_parse/data"
    reader.read_webpages(label_file_path, source_file_dir)
    for webpage in tqdm(reader):
        print(webpage.uid)
        # if(webpage.uid==916)
        # print(len(webpage.parent_list))
        # print(len(webpage.text_list))
        structure_dict, node_depth = dfp(webpage.parent_list)
        print(structure_dict)
        discourse_node_path = main(webpage.parent_list, webpage.previous_list)
        html_string = draw_html(discourse_node_path, node_depth, webpage.text_list)
        name = "golden" if webpage.golden else "predict"
        with open(os.path.join("html", f"newNEW_{webpage.uid}_{name}.htm"), "w", encoding="utf-8") as htm:
            htm.write(html_string)

    exit(10)
    # TODO: 还原图片 | 该图片来自微信公众平台未经允许不可引用
    # TODO: 加入Break or Continue的考虑 | √?
    # TODO: 合并Combine | TODO
    # TODO: 标注中的previous_node的定义并不完全等同于现在的规则!!!因为涉及到Node_Identity这个标签 | TODO
    # father958 = [-1, -1, -1, 2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 22, 0, 25, 0, 27, 27, 0, 30, 30, 30, 0, 34, -1, 36, 36, 36, 36, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # text958 = [str(i) for i in list(range(53))]
    # structure, depth, string = dfp(father958, text958)
    # print(structure)
    # print(depth)
    # print(string)

    """father = [-1, 0,1,1,0,-1]
    text = ["1", "2", "4", "5", "3", "6"]
    previous = [3, 3, 3, 0, 0, 0]
    previous = [3, 3, 3, 1, 1, 1]
    structure, depth, string = dfp(father, text, previous_list=previous)
    print(string)
    exit(10)"""


    reader = LabelReader()
    label_file_path = "robertawwmext.txt"
    source_file_dir = "../webpage_discourse_parse/data"
    reader.read_webpages(label_file_path, source_file_dir)
    for webpage in tqdm(reader):
        print(webpage.uid)
        # if(webpage.uid==916)
        # print(len(webpage.parent_list))
        # print(len(webpage.text_list))
        structure, depth, string = dfp(webpage.parent_list, webpage.text_list, previous_list=webpage.previous_list)
        name = "golden" if webpage.golden else "predict"
        with open(os.path.join("html", f"new6_{webpage.uid}_{name}.htm"), "w", encoding="utf-8") as htm:
            htm.write(HTM_PREFIX + string + HTM_SUFFIX)