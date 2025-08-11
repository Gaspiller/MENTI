import re
import time
import sys
import argparse
import json
import logging
import os  # 新增

from tqdm import tqdm

from Config import args
from MetaTool import MetaTool
from Configuration import Configuration
from Models import LLMs
from Prompts import Prompts

# ===== 日志部分改动：每次运行新文件 =====
os.makedirs("./logs", exist_ok=True)
log_filename = time.strftime("menti_process_%Y%m%d_%H%M%S.log")
log_path = os.path.join("./logs", log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_path,  # 使用带时间戳的新文件
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =====================================


class Agent:
    def __init__(self) -> None:
        pass

    def docagent(self, query: str, case: str, truth_scale: str = None) -> None:
        self.case = case
        self.query = query

        LLM = LLMs(args.llm_model)
        diagnose_result = LLM.generate(f"{Prompts().preliminary_diagnosis_prompt}\n\n{case}")

        self.agent_curation = MetaTool(query, diagnose_result)
        category, index, self.final_scale = self.agent_curation.execute()

        if truth_scale is not None and truth_scale != self.final_scale:
            self.final_result = 0
            return

        try:
            self.agent_configuration = Configuration(category, index, case)
            self.final_result = self.agent_configuration.execute()
        except:
            self.final_result = 0


if __name__ == "__main__":
    with open(args.case_path, encoding="UTF-8") as file:
        datas = json.loads(file.read())

    agent = Agent()
    agent.docagent(datas[args.eval_index]["doctor_query"], datas[args.eval_index]["patient_case"])
