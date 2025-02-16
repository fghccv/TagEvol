from FlagEmbedding import FlagICLModel
import os
import torch
import utils

def main():
    datas = utils.read_jsonl("/home/zhoushiqi/workplace/icaed/code/instag/datas/ori-datas/EvolInstruct-Code-80k.jsonl")
    instructions = [data['instruction'] for data in datas]
    
    examples = [
    {'instruct': 'Given a code-related question, retrieve relevant code that answers the question.',
    'query': 'Create a code snippet to substitute all instances of one word with another word in a given string in JavaScript.\nstring = \"This is a simple string\"',
    'response': "You can use the split() and join() methods in JavaScript to substitute all instances of one word with another word in a string. Here is an example:\n\n```javascript\nvar string = \"This is a simple string\";\nvar wordToReplace = \"simple\";\nvar replacementWord = \"new\";\n\nstring = string.split(wordToReplace).join(replacementWord);\nconsole.log(string);\n```\n\nIn this example, \"simple\" is replaced by \"new\", resulting in the string: \"This is a new string\". \n\nPlease note that this method is case-sensitive, which means it won't replace \"Simple\" with \"simple\". If you need this function to be case-insensitive, you'll need to use a regular expression with the \"i\" flag:\n\n```javascript\nvar string = \"This is a Simple string\";\nvar wordToReplace = \"simple\";\nvar replacementWord = \"new\";\n\nvar reg = new RegExp(wordToReplace,\"gi\"); \n\nstring = string.replace(reg, replacementWord);\nconsole.log(string);\n```\nIn this case, the \"i\" flag specifies that the regular expression should be case-insensitive. The \"g\" flag tells it to replace all matches, not just the first one. So, \"Simple\" is replaced by \"new\", resulting in the string: \"This is a new string\"."},
    {'instruct': 'Given a code-related question, retrieve relevant code that answers the question.',
    'query': 'How to create a C++ Boost undirected graph and traverse it in depth first search (DFS) order?',
    'response': "Let me illustrate this by creating a simple undirected graph using Boost Graph Library (BGL) and then traversing it using depth first search.\n\n1. First, include the necessary headers:\n\n```cpp\n#include <boost/graph/adjacency_list.hpp>\n#include <boost/graph/depth_first_search.hpp>\n```\n2. Define the graph type and necessary vertices:\n\n```cpp\ntypedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> Graph;\ntypedef boost::graph_traits<Graph>::vertex_descriptor Vertex;\n```\n3. Create your graph:\n\n```cpp\n// Create a graph object\nGraph g;\n\n// Add edges to the graph\nadd_edge(0, 1, g);\nadd_edge(1, 2, g);\nadd_edge(2, 3, g);\nadd_edge(3, 4, g);\nadd_edge(4, 0, g);\n```\nWe've created a simple 5-vertex cycle graph.\n\n4. Now, create your DFS visitor. In this case, a simple visitor that prints the name of each vertex as it's discovered:\n\n```cpp\nclass dfs_visitor : public boost::default_dfs_visitor\n{\npublic:\n    void discover_vertex(Vertex v, const Graph& g) const\n    {\n        std::cout << v << \" \";\n    }\n};\n```\n5. Finally, call depth_first_search with your graph and visitor:\n\n```cpp\ndfs_visitor vis;\nboost::depth_first_search(g, boost::visitor(vis));\n```\nWhen run, this will print the names of the vertices in the order they were discovered by the depth-first search. Please note that the actual output can vary because DFS does not always produce the same output for a given input graph, though it will always visit all vertices."}
    ]
    
    model = FlagICLModel('/home/zhoushiqi/workplace/model/bge-en-icl', 
                        # query_instruction_for_retrieval="Given a code-related question, retrieve relevant code that answers the question.",
                        query_instruction_for_retrieval="Represent this sentence for searching relevant codes:",
                        examples_for_task=None,  # set `examples_for_task=None` to use model without examples
                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    embeddings= model.encode_queries(instructions)
    print(embeddings.shape)
    torch.save(embeddings, "/home/zhoushiqi/workplace/icaed/embeddings/bge-en-icl/EvolInstruct-Code-80k-embedding.pth",pickle_protocol=4)
if __name__ == '__main__':
   main()