import argparse


class Config:
    def parseargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-lm', '--llm_model', type=str, default='gpt-4o')
        parser.add_argument('-em', '--embedding_model', type=str, default='m3e')

        parser.add_argument('-ei', '--eval_index', type=int, default=0)
        parser.add_argument('-t', '--test', action="store_true", help="Choose whether to print the results of each component.")

        parser.add_argument('-r', '--retry', action="store_false")
        parser.add_argument('-rn', '--retry_num', type=int, default=5)

        parser.add_argument('-p', '--parallel', action="store_false", help="Choose whether to parallelize the retrieval of Curation.")
        parser.add_argument('-pq', '--parallel_num_query', type=int, default=2)
        parser.add_argument('-pk', '--parallel_num_key', type=int, default=2)

        parser.add_argument('-dn', '--dispatch_num', type=int, default=10)

        parser.add_argument('--case_path', type=str, default="./CalcQA/clinical_case.json")
        parser.add_argument('--tool_scale_path', type=str, default="./CalcQA/tool_scale.json")
        parser.add_argument('--tool_unit_path', type=str, default="./CalcQA/tool_unit.json")

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()


args = Config()