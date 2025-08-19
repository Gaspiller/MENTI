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
            # 若只是缺参提示，直接返回提示，避免递归调用
            if isinstance(dcs_content, list) and all(isinstance(s, str) for s in dcs_content):
                if all(("请补充参数" in s) or ("缺少参数" in s) for s in dcs_content):
                    return "；".join(dcs_content)

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

        # 优先从 case 文本中的 calculator_parameters 代码块直接解析参数，成功则跳过 LLM 抽取
        try:
            m = re.search(r'calculator_parameters:\s*```json\s*(\{[\s\S]*?\})\s*```', self.case)
            if m:
                params = json.loads(m.group(1))
                return json.dumps(params, ensure_ascii=False)
        except Exception:
            pass

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

        # 仅对 APACHE II 执行硬规则；其它量表走 LLM 反射
        if self.tool.get("function_name") == "calculate_apache_ii_score":
            try:
                params = json.loads(self.parameter)

                def get_val(name):
                    item = params.get(name)
                    val = None if item is None else item.get("Value")
                    if isinstance(val, str) and val.strip().lower() == "null":
                        return None
                    if isinstance(val, str) and val.strip() == "":
                        return None
                    return val

                required_fields = [
                    "age", "temperature", "mean_arterial_pressure", "heart_rate", "respiratory_rate",
                    "sodium", "potassium", "creatinine", "hematocrit", "white_blood_cell_count",
                    "gcs", "ph", "history_of_severe_organ_insufficiency", "acute_renal_failure", "fio2"
                ]
                missing = [k for k in required_fields if get_val(k) is None]

                fio2 = get_val("fio2")
                pao2 = get_val("pao2")
                aagrad = get_val("a_a_gradient")

                fio2_is_high = None  # True: ≥50%，False: <50%
                if isinstance(fio2, (int, float)):
                    if fio2 in (0, 1):
                        fio2_is_high = (fio2 == 1)
                    else:
                        fio2_is_high = (float(fio2) >= 50.0)
                elif isinstance(fio2, str):
                    nums = re.findall(r"[\d\.]+", fio2)
                    fio2_is_high = (float(nums[0]) >= 50.0) if nums else None

                if fio2_is_high is True and aagrad is None:
                    missing.append("a_a_gradient")
                if fio2_is_high is False and pao2 is None:
                    missing.append("pao2")

                if missing:
                    return "toolcall", [f"请补充参数：{', '.join(sorted(set(missing)))}"]
                return "calculate", ""
            except Exception:
                pass

        configuration_reflect_prompt = self.prompts.configuration_reflect_exper_prompt
        configuration_reflect_input = configuration_reflect_prompt.replace("INSERT_DOC_HERE", self.tool["docstring"]).replace("INSERT_LIST_HERE", self.parameter)

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

        # 将 "null"/空串 标准化为 None，并只取 Value
        def norm(v):
            val = v.get("Value")
            if isinstance(val, str) and val.strip().lower() == "null":
                return None
            if isinstance(val, str) and val.strip() == "":
                return None
            return val
        arguments = {key: norm(value) for key, value in arguments.items()}

        # 仅 APACHE II 做 FiO2 规范与必填校验
        if self.tool.get("function_name") == "calculate_apache_ii_score":
            fio2 = arguments.get("fio2")
            if fio2 is not None:
                if isinstance(fio2, (int, float)):
                    if fio2 not in (0, 1):
                        arguments["fio2"] = 1 if float(fio2) >= 50 else 0
                elif isinstance(fio2, str):
                    nums = re.findall(r"[\d\.]+", fio2)
                    arguments["fio2"] = 1 if nums and float(nums[0]) >= 50 else 0

            required_fields = [
                "age", "temperature", "mean_arterial_pressure", "heart_rate", "respiratory_rate",
                "sodium", "potassium", "creatinine", "hematocrit", "white_blood_cell_count",
                "gcs", "ph", "history_of_severe_organ_insufficiency", "acute_renal_failure", "fio2"
            ]
            missing = [k for k in required_fields if arguments.get(k) is None]
            if arguments.get("fio2") == 1 and arguments.get("a_a_gradient") is None:
                missing.append("a_a_gradient")
            if arguments.get("fio2") == 0 and arguments.get("pao2") is None:
                missing.append("pao2")
            if missing:
                raise ValueError("缺少参数: " + ", ".join(sorted(set(missing))))

        ans = tool_call(**arguments)

        if args.test:
            logger.info(f"Calculated score/result: {ans}")

        return ans