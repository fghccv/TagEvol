
import json

def generate_tags_prompt(instruction):
    prompt = (
        "You are a tag annotator and need to assign tags to the given task.\n"
        f"# Task\n{instruction}\n\n"
        "Please follow the steps below to assign tags to the given task.\n"
        "Step 1: Consider from which aspects that tags can be assigned to cover all features of the task as comprehensively as possible and provide a brief explanation. Aspects need to be **diverse** and **summarizing**, such as 'Task Type', 'Complexity'.etc.\n"
        "Step 2: Based on the #Aspect List# obtained in Step 1, assign tags to the given task from each aspect.\n"
        "Please reply strictly in the following format:\n"
        "step 1  #Aspect List and Explanation#:\n"
        "step 2 #Aspect2Tags#:\n"
        '#Aspect2Tags\n```{"xxx":[tag1, tag2, ...], "xxx":[tag1, tag2, ...], ...}```\nwhere xxx means aspect you get in step 1'
    )
    return prompt


def generate_instruction_prompt(data_point):
    shot_tags, shot_instructions, sub_tags = data_point['shot_tags'], data_point['shot_instructions'], data_point['sub_tags']
    prompt = (
        "You are an instruction generator capable of creating high-quality, challenging math-related instructions based on the given #Tags#.\n"
        "Please imitate the way humans ask questions, generate a high-quality, challenging math-related instruction based on the given #Tags#.\n"
        "Just provide the #Instruction# without any explanation.\n"
        "The #Instruction# MUST be reasonable and solvable.\n"
    )
    for i, (tags, instruction) in enumerate(zip(shot_tags, shot_instructions)):
        prompt += f"#Tags#:\n{json.dumps(tags)}\n"
        prompt += f"#Instruction#:\n{instruction}\n\n"
    prompt += f"#Tags#:\n{json.dumps(sub_tags)}\n"
    prompt += "#Instruction#:"
    # print(prompt)
    # input()
    return prompt


def generate_instruction_zero_prompt(data_point):
    sub_tags = data_point['sub_tags']
    prompt = (
        "You are an instruction generator capable of creating high-quality, challenging math-related instructions based on the given #Tags#.\n"
        "Please imitate the way humans ask questions, generate a high-quality, challenging math-related instruction based on the given #Tags#.\n"
        "Just provide the #Instruction# without any explanation.\n"
        "The #Instruction# MUST be reasonable and solvable.\n"
        "Please reply strictly in the following format:\n\n"
        "#Instruction#:...\n"  
        "Below is the #Tags#:\n"
    )

    prompt += f"#Tags#:\n{json.dumps(sub_tags)}\n"
    prompt += "#Instruction#:"
    # print(prompt)
    # input()
    return prompt

def generate_reduce_prompt(merge_dict):
    prompt = (
        "You are a tag annotation expert capable of performing operations on the given #Tag Dictionary#.\n"
        "Please follow the steps below to deduplicate and expand the given #Tag Dictionary#."
        "In the #Tag Dictionary#, the keys represent tag categories, while the values are the corresponding tags.\n"
        "Step 1: Please carefully read #Tag Dictionary#, merge keys in the #Tag Dictionary# that have similar meanings (such as programming language and langauge), then for each merged key, merge values (i.e., tags) with similar meanings."
        "Please ensure that each key and value in the  #Deduplicated Tag Dictionary#  have distinct meanings.\n"
        "Step 2: Please expand the tags within each key in the #Deduplicated Tag Dictionary# obtained from Step 1."
        "Try to add 1-2 new tags to each key while ensuring they MUST be distinct from the existing tags.\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Deduplicated Tag Dictionary#:```...```\n"
        "Step 2 #Finally Expanded Tag Dictionary#:```...```\n\n"
        f"Below is the #Tag Dictionary#:\n```\n{json.dumps(merge_dict, indent=2)}\n```\n"

    )
    return prompt

def generate_evol_prompt(tags, ori_instruction):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tag List#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please read the #Instruction# and #Tag List# carefully and select a subset from #Tag List# so that #Instruction# can evolve based on these tags to generate a more difficult and high-quality version"
        "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
        # "The length of the subset is 1."
        "Do not select the tags included in #Instruction#."
        "Step 2: Please create a comprehensive plan based on the #Tag subset# generated in Step 1 to make the #Instruction# "
        "more challenging. The plan should include several tags from the #Tag subset#.\n\n"
        "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
        "add 10 to 20 words into the #Instruction#.\n\n"
        "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Tag subset:\n"
        "Step 2 #Plan#:\n"
        "Step 3 #Rewritten Instruction#:\n"
        "Step 4 #Finally Rewritten Instruction#:\n\n"
    )
    return prompt

def generate_cgtags_prompt(instruction):
    prompt = (
        "You are a tagging system that provides useful tags for task intentions to distinguish task for a helpful AI assistant.\n"
        f"# Task\n{instruction}\n\n"
        "Please follow the steps below to assign tags to the given task.\n"
        "Step 1: Consider from which aspects that tags can be assigned to cover main features of the task and provide a brief explanation. "
        "Aspects need to be **summarizing**, such as 'Required skill'.etc."
        "Please summarize this task with as few aspects as possible.\n"
        "Step 2: Based on the #Aspect List# obtained in Step 1, assign core tags to the given task from each aspect.\n"
        "Please reply strictly in the following format:\n"
        "step 1  #Aspect List and Explanation#:\n"
        "step 2 #Aspect2Tags#:\n"
        '#Aspect2Tags\n```{"xxx":[tag1, tag2, ...], "xxx":[tag1, tag2, ...], ...}```\nwhere xxx means aspect you get in step 1. Please response in English.'

    )
    return prompt



def generate_evol_random_prompt(tags, ori_instruction):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tags#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please create a comprehensive plan based on the #Tags# to make the #Instruction# "
        "more challenging (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle)."
        "The plan should include tags from the #Tags#.\n\n"
        "Step 2: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
        "add 10 to 20 words into the #Instruction#.\n\n"
        "Step 3: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Plan#:\n"
        "Step 2 #Rewritten Instruction#:\n"
        "Step 3 #Finally Rewritten Instruction#:\n\n"
    )
    return prompt


def generate_evol_allowback_prompt(tags, ori_instruction, ori_tags):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tags of Instruction#:\n{ori_tags}\n"
        f"Here is the #New Tag List#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please read the #Instruction# and #New Tags List# carefully and select a subset from #New Tag List# so that #Instruction# can evolve based on these tags to generate a more difficult and high-quality version"
        "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
        "The length of the subset is 1."
        "Do not select the similar tags included in #Tags of Instruction#."
        "Step 2: Please create a comprehensive plan based on the #Tag subset# generated in Step 1 to make the #Instruction# "
        "more challenging. The plan should include several tags from the #New Tag subset#.\n\n"
        "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
        "add 10 to 20 words into the #Instruction#."
        "Ensure that the #Rewritten Instruction# is reasonable and sovable.\n\n"
        # "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        # "Check whether the values in #Rewritten Instruction# are appropriate. Ensure that the #Rewritten Instruction# is reasonable and sovable. "
        "Step 4: If the #Rewritten Instruction# is not reasonable, back to the #Instruction#."
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Tag subset:\n"
        "Step 2 #Plan#:\n"
        "Step 3 #Rewritten Instruction#:\n"
        "Step 4 #Final Rewritten Instruction#:\n\n"
    )
    return prompt

def generate_evol_no_length_limited_prompt(tags, ori_instruction):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tag List#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please read the #Instruction# and #Tag List# carefully and select a subset from #Tag List# so that #Instruction# can evolve based on these tags to generate a more difficult and high-quality version"
        "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
        "The length of the subset is 1."
        "Do not select the tags included in #Instruction#."
        "Step 2: Please create a comprehensive plan based on the #Tag subset# generated in Step 1 to make the #Instruction# "
        "more challenging. The plan should include several tags from the #Tag subset#.\n\n"
        "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#.\n\n"
        "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Tag subset:\n"
        "Step 2 #Plan#:\n"
        "Step 3 #Rewritten Instruction#:\n"
        "Step 4 #Finally Rewritten Instruction#:\n\n"
    )
    return prompt

def generate_evol_prompt_multitag(tags, ori_instruction, num_tag):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tag List#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please read the #Instruction# and #Tag List# carefully and select a subset from #Tag List# so that #Instruction# can evolve based on these tags to generate a more difficult and high-quality version"
        "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
        f"The length of the subset is {num_tag}."
        "Do not select the tags included in #Instruction#."
        "Step 2: Please create a comprehensive plan based on the #Tag subset# generated in Step 1 to make the #Instruction# "
        "more challenging. The plan should include several tags from the #Tag subset#.\n\n"
        "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
        "add 10 to 20 words into the #Instruction#.\n\n"
        "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Tag subset:\n"
        "Step 2 #Plan#:\n"
        "Step 3 #Rewritten Instruction#:\n"
        "Step 4 #Finally Rewritten Instruction#:\n\n"
    )
    return prompt
def generate_nokeytags_prompt(instruction):
    prompt = (
        "You are a tagging system that provides useful tags for instruction intentions"
        " to distinguish instructions for a helpful AI assistant. Below is an instruction:\n"
        f"[begin]\n{instruction}\n[end]\n"
        "Please provide fine-grained tags to identify main intentions of above instruction.\n"
        "Your answer should be a list including titles of tags and a brief explanation of each tag."
        " Your response have to strictly follow this JSON format:\n"
        "```[{\"tag\": str, \"explanation\": str}]```. Please response in English."
    )
    return prompt

# def generate_evol_prompt_multitag_20mul3up(tags, ori_instruction, num_tag):
#     prompt = (
#         "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
#         f"Here is the #Instruction#:\n{ori_instruction}\n"
#         f"Here is the #Tag List#:\n{tags}\n\n"
#         "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
#         "Step 1: Please read the #Instruction# and #Tag List# carefully and select a subset from #Tag List# so that #Instruction# can evolve based on these tags to generate a more difficult and high-quality version"
#         "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
#         f"The length of the subset is {num_tag}."
#         "Do not select the tags included in #Instruction#."
#         "Step 2: Please create a comprehensive plan based on the #Tag subset# generated in Step 1 to make the #Instruction# "
#         "more challenging. The plan should include several tags from the #Tag subset#.\n\n"
#         "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
#         f"add 10 to {20*num_tag} words into the #Instruction#.\n\n"
#         "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
#         "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
#         "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
#         "Please reply strictly in the following format:\n\n"
#         "Step 1 #Tag subset#:\n"
#         "Step 2 #Plan#:\n"
#         "Step 3 #Rewritten Instruction#:\n"
#         "Step 4 #Finally Rewritten Instruction#:\n\n"
#     )
#     return prompt

def generate_evol_prompt_multitag_20mul3up(tags, ori_instruction, num_tag):
    prompt = (
    "You are an Instruction Rewriter tasked with transforming the provided #Instruction# into a more challenging version based on the specified tags.\n\n"
    f"Here is the #Instruction#:\n{ori_instruction}\n\n"
    f"Here is the #Tag List#:\n{tags}\n\n"
    "Please follow the steps below to rewrite the given #Instruction# into a more intricate and demanding version.\n\n"
    "Step 1: Carefully read the #Instruction# and #Tag List#. Select a subset from the #Tag List# that will allow the #Instruction# to evolve and become more complex. "
    "The chosen subset should provide richer, more nuanced information that enhances the original prompt, ultimately increasing its difficulty and quality. "
    f"Aim to challenge advanced AI assistants like ChatGPT and GPT-4. The subset should contain {num_tag} tags and must exclude any tags already present in the #Instruction#.\n\n"
    "Step 2: Develop a comprehensive enhancement plan using the #Tag subset# chosen in Step 1. "
    "This plan should focus on seamlessly integrating multiple tags from the #Tag subset# into the original #Instruction#. "
    "The incorporation should feel natural and elevate the complexity of the task, ensuring that the revised instruction poses a greater challenge while maintaining coherence.\n\n"
    f"Step 3: Execute the plan step by step and provide the #Rewritten Instruction#. The #Rewritten Instruction# may only add between 10 and {20*num_tag} words to the original #Instruction#.\n\n"
    "Step 4: Thoroughly review the #Rewritten Instruction# to identify any inconsistencies. "
    "Ensure that the #Rewritten Instruction# is solely a more challenging version of the original #Instruction#. "
    "Provide only the #Final Rewritten Instruction# without any additional explanation.\n\n"
    "Please respond strictly in the following format:\n\n"
    "Step 1 #Tag subset#:\n"
    "Step 2 #Plan#:\n"
    "Step 3 #Rewritten Instruction#:\n"
    "Step 4 #Finally Rewritten Instruction#:\n"
)
    return prompt

def generate_evol_prompt_multitag_20mul3up_yxprompt2(merge_tags, ori_instruction, ori_tags, num_tag):
    prompt = (
    "You are an Instruction generator. Based on the following Tag set:\n"
    f"{json.dumps(ori_tags)}\n"
    "you need to:\n"
    f"Step 1: Select {num_tag} Tags from the list below to augment the tag set:\n"
    f"{json.dumps(merge_tags)}\n\n"
    "Step 2: Merge the Tag set and analyze how to use the merged Tag set to generate a challenging instruction from scratch.\n\n"
    "Step 3: Execute the Plan from the previous step and generate an instruction based on the Tag set.\n\n"
    f"Instruction Format: <begin>{ori_instruction}<end>\n\n"
    "Strictly follow the format below:\n"
    "Step1#Selected Tag#:\n"
    "Step2#Plan#:\n"
    "Step3#Instruction#:<begin>Instruction<end>\n"
)
    return prompt
def generate_evol_prompt_multitag_20mul3up_random(tags, ori_instruction, num_tag):
    prompt = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more challenging version based on the given tags.\n"
        f"Here is the #Instruction#:\n{ori_instruction}\n"
        f"Here is the #Tag List#:\n{tags}\n\n"
        "Please follow the steps below to rewrite the given #Instruction# into a more challenging version.\n\n"
        "Step 1: Please create a comprehensive plan based on the #Tag List# to make the #Instruction# more challenging."
        "(to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle)."
        " The plan should include all tags from the #Tag List#.\n\n"
        "Step 2: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
        f"add 10 to {20*num_tag} words into the #Instruction#.\n\n"
        "Step 3: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
        "Ensure that the #Rewritten Instruction# is only a more challenging version of the #Instruction#. "
        "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
        "Please reply strictly in the following format:\n\n"
        "Step 1 #Plan#:\n"
        "Step 2 #Rewritten Instruction#:\n"
        "Step 3 #Finally Rewritten Instruction#:\n\n"
    )
    return prompt