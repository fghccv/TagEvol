from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import asyncio

def construct_prompt(inst):
  PROMPT = (
    "You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction:\n"
    "[begin]\n"
    f"{inst}\n"
    "[end]\n"
    "Please provide coarse-grained tags, such as 'Spelling and Grammar Check' and 'Cosplay', to identify main intentions of above instruction.\n"
    "Your answer should be a list including titles of tags and a brief explanation of each tag. Your response has to strictly follow this JSON format:\n"
    "```[{\"tag\": str, \"explanation\": str}]```. Please respond in English."
  )
  return PROMPT
async def async_query_openai(query, idx):
  aclient = AsyncOpenAI(
      base_url="https://one.ooo.cool/v1",
      api_key="sk-E50mKr6L2k5gPrYu48A2BeEa8c75484cB36aD2EeA334231d"
  )
  
  while True:
    try:
        completion = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
        )
        break
    except Exception as e:
        print(e)
        await asyncio.sleep(1) 
  return (idx, completion.choices[0].message.content)

async def async_process_queries(queries, idxs):
    tasks = [async_query_openai(query, idx) for idx, query in zip(idxs, queries)]
    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying OpenAI"):
        result = await future  # 等待任务完成并获取结果
        results.append(result)
    
    return results   

async def main():
  import json,random,re,tqdm
  random.seed(42)
  datas = json.load(open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math+gsm8k/tag_reduce_new/datas/final_datas/all_math_15k_72bres.json"))
  id2datas = {data['id']:data for data in datas}
  sample_ids = random.sample(list(range(14000)), k=50)
  sample_datas = [id2datas[id] for id in sample_ids]
  all_scores = []
  prompts = [construct_prompt(data['instruction']) for data in sample_datas]
  results = await async_process_queries(prompts, sample_ids)
  id2tags={}
  for res in results:
    try:
      tags = re.findall("```(.+)```",res[1],re.DOTALL)[0].strip("json").strip()
      # print(tags)
      tags = json.loads(tags)
      # print(tags)
      tags = [t['tag'] for t in tags]
      id2tags[res[0]] = tags
      all_scores.append(len(tags))
    except Exception as e:
      print(e)
      continue
  json.dump(id2tags, open("/home/zhoushiqi/workplace/TagReduce/TagReduce-math+gsm8k/temp.json", "w"))
  print(sum(all_scores)/len(all_scores))
if __name__ == "__main__":
  asyncio.run(main())

