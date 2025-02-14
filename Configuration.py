import json
import re
import math
import datetime
import logging

from Models import LLMs, Embeddings
from Prompts import Prompts
from MetaTool import MetaTool
from Config import args


logger = logging.getLogger(__name__)


class Configuration:
    def __init__(self, tool_category: str, tool_index: int, case: str=None) -> None:
        self.tool_category = tool_category
        self.tool_index = tool_index
        self.case = case
        self.llm_model_name = args.llm_model
        self.embedding_model_name = args.embedding_model

        self.llm_model = LLMs(args.llm_model)
        self.prompts = Prompts()

        if self.tool_category == "scale":
            with open(args.tool_scale_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())
        elif self.tool_category == "unit":
            with open(args.tool_unit_path, encoding="UTF-8") as file:
                self.tool_list = json.loads(file.read())
        self.tool = self.tool_list[self.tool_index]

    def execute(self) -> int | str:
        self.parameter = self.extract()
        dcs_category, dcs_content = self.reflect()

        if dcs_category == "calculate":
            ans = self.calculate()
            return ans
        elif dcs_category == "toolcall":
            for subtask in dcs_content:
                # toolcalling
                metatool = MetaTool(subtask, "")
                category, index, _ = metatool.execute()
                configuration = Configuration(category, index, subtask)
                tool_ans = configuration.execute()
                self.case = f"{self.case}\n\n{tool_ans}"

            ans = self.execute()

        return ans

    def extract(self) -> str:
        if args.test:
            logger.info("====================Now in Extract====================")

        configuration_extract_prompt = self.prompts.configuration_extract_prompt
        configuration_extract_input = configuration_extract_prompt.replace("INSERT_DOCSTRING_HERE", self.tool["docstring"]).replace("INSERT_TEXT_HERE", self.case)
        
        LLM = self.llm_model

        task_completed = False
        cnt = 0
        while task_completed == False:
            if cnt >= args.retry_num:
                raise ValueError("Extract Error.")
            try:
                ans = LLM.generate(configuration_extract_input)
                if args.test:
                    logger.info(ans)
                ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                task_completed = True
            except:
                cnt += 1

        return ans

    def reflect(self) -> tuple[str, str]:
        if args.test:
            logger.info("====================Now in Reflect====================")

        if self.tool_category == "unit":
            return "calculate", ""
        
        configuration_reflect_prompt = self.prompts.configuration_reflect_exper_prompt
        configuration_reflect_input = configuration_reflect_prompt.replace("INSERT_DOC_HERE",self.tool["docstring"]).replace("INSERT_LIST_HERE", self.parameter)

        LLM = self.llm_model

        task_completed = False
        cnt = 0
        while task_completed == False:
            if cnt >= args.retry_num:
                raise ValueError("Reflect Error.")
            try:
                ans = LLM.generate(configuration_reflect_input)
                if args.test:
                    logger.info(ans)
                ans = re.findall(r"```json(.*?)```", ans, flags=re.DOTALL)[0].strip()
                ans = json.loads(ans)
                task_completed = True
            except:
                cnt += 1
        
        category = ans["chosen_decision_name"]
        content = ans["supplementary_information"]

        return category, content

    def calculate(self) -> int | str:
        if args.test:
            logger.info("====================Now in Calculate====================")

        code = self.tool["code"]
        exec(code)

        if args.test:
            logger.info(self.tool["function_name"])
        tool_call = locals()[self.tool["function_name"]]
        arguments = json.loads(self.parameter)
        arguments = {key: value["Value"] for key, value in arguments.items()}
        try:
            ans = tool_call(**arguments)
        except Exception as e:
            logger.info("Calculate Error:", e)

        if args.test:
            logger.info(ans)

        return ans
