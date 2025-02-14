import re
import time
import json
import logging

import numpy as np
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed

from Models import LLMs, Embeddings
from Prompts import Prompts
from Config import args


logger = logging.getLogger(__name__)


class MetaTool:
    def __init__(self, query: str , scenario: str="") -> None:
        self.query = query
        self.scenario = scenario
        self.llm_model_name = args.llm_model
        self.embedding_model_name = args.embedding_model

        self.llm_model = LLMs(args.llm_model)
        self.embedding_model = Embeddings(args.embedding_model).embedding_model
        self.prompts = Prompts()


    def execute(self) -> tuple[str, int, str]:
        self.toolkit = self.classify()

        self.query_rewrited = self.rewrite()
        
        if args.parallel:
            self.ranking = self.retrieve_parallel()
        else:
            self.ranking = self.retrieve_nonparallel()

        self.topindex = self.toolindex()

        self.finalans, self.finalname = self.dispatch()

        return (self.toolkit, self.finalans, self.finalname)


    def classify(self) -> str:
        if args.test:
            logger.info("====================Now in Classify====================")

        LLM = self.llm_model
        metatool_classify_prompt = self.prompts.metatool_classify_prompt
        metatool_classify_input = metatool_classify_prompt.replace("INSERT_QUERY_HERE", self.query)
        
        task_completed = False
        cnt = 0
        while task_completed == False:
            if cnt >= args.retry_num:
                raise ValueError("Classify Error.")
            try:
                ans = LLM.generate(metatool_classify_input)
                ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                if args.test:
                    logger.info(ans)
                metatool_classify = json.loads(ans)
                task_completed = True
            except:
                cnt += 1

        return metatool_classify["chosen_toolkit_name"]


    def rewrite(self) -> list[str]:
        if args.test:
            logger.info("====================Now in Rewrite====================")
    
        if self.scenario != "":
            metatool_rewrite_prompt = self.prompts.metatool_rewrite_prompt_withtext
            metatool_rewrite_input = metatool_rewrite_prompt.replace('INSERT_CASE_HERE', self.scenario).replace('INSERT_QUERY_HERE', self.query)

            LLM = self.llm_model

            task_completed = False
            cnt = 0
            while task_completed == False:
                if cnt >= args.retry_num:
                    raise ValueError("Rewrite Error.")
                try:
                    ans = LLM.generate(metatool_rewrite_input)
                    ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                    if args.test:
                        logger.info(ans)
                    ans = json.loads(ans)
                    task_completed = True
                except:
                    cnt += 1

        elif self.scenario == "":
            metatool_rewrite_prompt = self.prompts.metatool_rewrite_prompt_outtext
            metatool_rewrite_input = metatool_rewrite_prompt.replace('INSERT_QUERY_HERE', self.query)

            LLM = self.llm_model

            task_completed = False
            cnt = 0
            while task_completed == False:
                if cnt>=5:
                    raise ValueError("Rewrite Error.")
                try:
                    ans = LLM.generate(metatool_rewrite_input)
                    ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                    if args.test:
                        logger.info(ans)
                    ans = json.loads(ans)
                    task_completed = True
                except:
                    cnt += 1

        return ans
    

    @staticmethod
    def cosine_similarity(embed_a: np.ndarray, embed_b: np.ndarray) -> float:
        dot_product = np.dot(embed_a, embed_b)
        cosine = dot_product / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b))
        return cosine

    def ranking(self, embed_demand: np.ndarray, embed_list: np.ndarray) -> list[int]:
        retrieve_list = [self.cosine_similarity(embed_demand, embed) for embed in embed_list]

        tem_retrieve_list = [(index, value) for index, value in enumerate(retrieve_list)]
        sort_list = sorted(tem_retrieve_list, key=lambda x: x[1], reverse=True)

        rank = [0 for _ in retrieve_list]
        for index, value in enumerate(sort_list):
            rank[value[0]] = index + 1

        return rank

    
    def retrieve_nonparallel(self) -> list[int]:
        '''
            return the rank of each tools.
        '''

        if self.toolkit == "scale":
            with open(args.tool_scale_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())
        elif self.toolkit == "unit":
            with open(args.tool_unit_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())

        if args.test:
            logger.info("====================Now in Retrieve====================")
            start = time.time()

        rank_query_list = []
        for demand in self.query_rewrited:
            embed_demand = self.embedding_model.encode(demand)

            rank_key_list = []
            embed_list_1 = self.embedding_model.encode([f'''{tool["function_name"]}\n\n{tool["tool_name"]}''' for tool in self.tool_list])
            rank_key_list.append(self.ranking(embed_demand, embed_list_1))

            embed_list_2 = self.embedding_model.encode([f'''{tool["function_name"]}\n\n{tool["tool_name"]}\n\n{tool["docstring"]}''' for tool in self.tool_list])
            rank_key_list.append(self.ranking(embed_demand, embed_list_2))

            embed_list_3 = self.embedding_model.encode([f'''{tool["function_name"]}\n\n{tool["tool_name"]}\n\n{tool["description"]}''' for tool in self.tool_list])
            rank_key_list.append(self.ranking(embed_demand, embed_list_3))

            rank = self.rerank_rrf(rank_key_list)
            rank_query_list.append(rank)

        rank = self.rerank_rrf(rank_query_list)
        # self.result_present_tool.append(rank)
        if args.test:
            logger.info(f"Retrieve cost time: {time.time()-start}")
            logger.info(rank)
        return rank

    def retrieve_multi_key(self, embed_demand: np.ndarray, key_list: list[str]) -> list[int]:
        embed_list = self.embedding_model.encode(key_list)
        rank = self.ranking(embed_demand, embed_list)

        return rank

    def retrieve_multi_demand(self, demand: str) -> list[int]:
        embed_demand = self.embedding_model.encode(demand)

        key_list_1 = [f'''{tool["function_name"]}\n\n{tool["tool_name"]}''' for tool in self.tool_list]
        key_list_2 = [f'''{tool["function_name"]}\n\n{tool["tool_name"]}\n\n{tool["docstring"]}''' for tool in self.tool_list] # 等有了docstring用docstring
        key_list_3 = [f'''{tool["function_name"]}\n\n{tool["tool_name"]}\n\n{tool["description"]}''' for tool in self.tool_list]

        rank_key_list = []
        with ThreadPoolExecutor(max_workers=args.parallel_num_key) as thread_pool_key:
            f1 = thread_pool_key.submit(self.retrieve_multi_key, embed_demand, key_list_1)
            f2 = thread_pool_key.submit(self.retrieve_multi_key, embed_demand, key_list_2)
            f3 = thread_pool_key.submit(self.retrieve_multi_key, embed_demand, key_list_3)
            rank_key_list.append(f1.result())
            rank_key_list.append(f2.result())
            rank_key_list.append(f3.result())

        rank = self.rerank_rrf(rank_key_list)

        return rank

    def retrieve_parallel(self) -> list[int]:

        if self.toolkit == "scale":
            with open(args.tool_scale_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())
        elif self.toolkit == "unit":
            with open(args.tool_unit_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())

        if args.test:
            logger.info("====================Now in Retrieve====================")
            start = time.time()

        with ThreadPoolExecutor(max_workers=args.parallel_num_query) as thread_pool_demand:
            results = [thread_pool_demand.submit(self.retrieve_multi_demand, demand) for demand in self.query_rewrited]
            rank_query_list = [f.result() for f in as_completed(results)]

        rank = self.rerank_rrf(rank_query_list)
        # self.result_present_tool.append(rank)
        if args.test:
            logger.info(f"Retrieve cost time: {time.time()-start}")
            logger.info(rank)
        return rank
    
    def rerank_rrf(self, rank_list: list[list[int]], rrf_k: list[int] = None) -> list[int]:
        '''
            Combine ranking sequences using rrf.
        '''

        rerank_num = len(rank_list)
        list_num = len(rank_list[0])

        if rrf_k is None:
            rrf_k = [60] * rerank_num
        score_list = [0] * list_num

        for index, l in enumerate(rank_list):
            for i, j in enumerate(l):
                score_list[i] += 1 / (j + rrf_k[index])

        tem_score_list = [(index, value) for index, value in enumerate(score_list)]
        sort_list = sorted(tem_score_list, key=lambda x: x[1], reverse=True)

        rank = [0 for _ in score_list]
        for index, value in enumerate(sort_list):
            rank[value[0]] = index + 1

        return rank


    def toolindex(self) -> list[int]:

        tem_ranking = [(index, value) for index, value in enumerate(self.ranking)]
        top = sorted(tem_ranking, key=lambda x: x[1])
        top_index = [value[0] for value in top]
        if args.test:
            logger.info(top_index)

        return top_index
    
    def dispatch(self):
        if args.test:
            logger.info("====================Now in Dispatch====================")

        if self.toolkit == "scale":
            with open(args.tool_scale_path, encoding="UTF-8") as file:
                tool_list = json.loads(file.read())
        elif self.toolkit == "unit":
            with open(args.tool_unit_path, encoding="UTF-8") as file:
                tool_list = json.loads(file.read())

        retrieve_tool_name = []
        retrieve_tool_inst = ""
        for i in self.topindex[:args.dispatch_num]:
            if args.test:
                logger.info(tool_list[i]["tool_name"])
            retrieve_tool_name.append(tool_list[i]["tool_name"])
            retrieve_tool_inst += f'''{tool_list[i]["tool_name"]}: {tool_list[i]["docstring"]}\n\n'''

        metatool_dispatch_prompt = self.prompts.metatool_dispatch_prompt
        metatool_dispatch_input = metatool_dispatch_prompt.replace('INSERT_DEMAND_HERE', self.query).replace('INSERT_TOOLLIST_HERE', str(retrieve_tool_name))
        metatool_dispatch_input = metatool_dispatch_input.replace('INSERT_TOOLINST_HERE', retrieve_tool_inst).replace('INSERT_SCE_HERE', self.scenario)

        LLM = self.llm_model

        task_completed = False
        cnt = 0
        while task_completed == False:
            if cnt >= args.retry_num:
                raise ValueError("Dispatch Error.")
            try:
                ans = LLM.generate(metatool_dispatch_input)
                if args.test:
                    logger.info(ans)
                ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                ans = json.loads(ans)
                ans = ans["chosen_tool_name"].lower()
                task_completed = True
            except:
                cnt += 1

        if args.test:
            logger.info(ans)

        for i in self.topindex[:args.dispatch_num]:
            if tool_list[i]["tool_name"].lower() == ans:
                if args.test:
                    logger.info(i)
                return i, tool_list[i]["tool_name"]

        raise ValueError("Dispatch Error.")