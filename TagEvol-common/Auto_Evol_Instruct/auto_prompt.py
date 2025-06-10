# Base instruction template for generating new prompts
BASE_INSTRUCTION = (
    "You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.\n"
    "Please follow the steps below to rewrite the given #Instruction# into a more complex version.\n\n"
    "Step 1: Carefully read the initial instruction and identify all the elements involved - this includes variables, constants, operations, and conditions.\n"
    "Step 2: Consider how each element could be made more complex. For variables, this could involve introducing more variables or making the existing variables dependent on others. For constants, consider changing them to variables or making them dependent on other factors. For operations, consider introducing more complex operations or multiple steps. For conditions, consider adding more conditions or making the existing conditions more complex.\n"
    "Step 3: Formulate a plan to integrate these complexities into the instruction. Ensure that the changes are coherent and relevant to the initial problem context. The plan should not just randomly add complexity but should make the problem more interesting or challenging in a meaningful way. Avoid introducing irrelevant concepts or complicating the problem to the extent of changing its nature.\n"
    "Step 4: Rewrite the instruction according to the plan. Ensure that the rewritten instruction is still understandable and that it accurately represents the initial problem context. The rewritten instruction should only add 10 to 20 words to the original instruction. Make sure that the progression of complexity is smooth and gradual.\n"
    "Step 5: Review the rewritten instruction and check for any inaccuracies or inconsistencies. Make sure that the rewritten instruction is a more complex version of the original instruction and not a completely different problem. If any parts of the rewritten instruction are unreasonable or do not fit the problem context, revise them as necessary.\n"
    "Step 6: Ensure that the complexity increase is consistent and logical. Avoid introducing new conditions or variables that are not related to the initial problem. The complexity should evolve from the initial problem and not transform it into a different problem.\n"
    "Step 7: Test the rewritten instruction to ensure that it is solvable and that the complexity has indeed increased. If the problem is too difficult or impossible to solve, revise it as necessary.\n"
    "Please reply strictly in the following format:\n\n"
    "Step 1 #Elements Identified#:\n"
    "Step 2 #Complexity Additions#:\n"
    "Step 3 #Plan#:\n"
    "Step 4 #Rewritten Instruction#:\n"
    "Step 5 #Revised Instruction#:\n"
    "Step 6 #Consistency Check#:\n"
    "Step 7 #Final Rewritten Instruction#:\n\n"
    "#Instruction#:\n"
    "{}"
)
# BASE_INSTRUCTION = (
#     "You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.\n"
#     "Please follow the steps below to rewrite the given #Instruction# into a more complex version.\n\n"
#     "Step 1: Please read the #Instruction# carefully and list all the possible methods to "
#     "make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
#     "Please do not provide methods to change the language of the instruction!\n\n"
#     "Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# "
#     "more complex. The plan should include several methods from the #Methods List#.\n\n"
#     "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
#     "add 10 to 20 words into the #Instruction#.\n\n"
#     "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
#     "Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. "
#     "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
#     "Please reply strictly in the following format:\n\n"
#     "Step 1 #Methods List#:\n"
#     "Step 2 #Plan#:\n"
#     "Step 3 #Rewritten Instruction#:\n"
#     "Step 4 #Finally Rewritten Instruction#:\n\n"
#     "#Instruction#:\n"
#     "{}"
# )
def create_auto_prompt(instruction: str) -> str:
    """
    Generates a new prompt based on the given instruction by drawing inspiration
    from the base instruction and adding the provided instruction as the given prompt.

    Args:
        instruction (str): The given prompt from which a new prompt is generated.

    Returns:
        str: The complete prompt ready for use.
    """
    # Construct the prompt by combining the base instruction with the given instruction
    new_prompt = BASE_INSTRUCTION.format(instruction)
    
    return new_prompt
