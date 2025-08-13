#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeNTi 工具选择和分数正确率评估脚本
比较标准答案（clinical_case.json）与模型输出结果（logs-gpt-3.5/*.log）
"""

import json
import os
import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MentiEvaluator:
    def __init__(self, standard_file: str = "CalcQA/clinical_case.json", 
                 log_dir: str = "logs-gpt-3.5"):
        """
        初始化评估器
        
        Args:
            standard_file: 标准答案文件路径
            log_dir: 日志文件目录
        """
        self.standard_file = standard_file
        self.log_dir = log_dir
        self.standard_data = {}
        self.results = {
            'tool_accuracy': 0.0,
            'score_accuracy': 0.0,
            'total_cases': 0,
            'tool_correct': 0,
            'score_correct': 0,
            'detailed_results': [],
            'missing_logs': [],
            'error_logs': [],
            'tool_error_indices': [],  # 工具选择错误的索引
            'score_error_indices': []  # 分数计算错误的索引
        }
        
    def load_standard_data(self):
        """加载标准答案数据"""
        try:
            logger.info(f"正在加载标准答案文件: {self.standard_file}")
            with open(self.standard_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 按index建立索引
            for item in data:
                index = item.get('index')
                if index is not None:
                    self.standard_data[index] = {
                        'calculator_name': item.get('calculator_name', '').strip(),
                        'calculator_score': item.get('calculator_score'),
                        'doctor_query': item.get('doctor_query', ''),
                        'patient_case': item.get('patient_case', ''),
                    }
            
            logger.info(f"成功加载 {len(self.standard_data)} 条标准答案")
            
        except Exception as e:
            logger.error(f"加载标准答案失败: {e}")
            raise
    
    def extract_tool_name_from_log(self, log_content: str) -> Optional[str]:
        """
        从日志内容中提取选择的工具名称
        
        Args:
            log_content: 日志文件内容
            
        Returns:
            提取到的工具名称，如果未找到则返回None
        """
        # 优先查找最终选择的工具（scale类型）
        scale_pattern = r'"chosen_tool_name":\s*"([^"]+)"'
        scale_matches = re.findall(scale_pattern, log_content)
        
        # 过滤出医疗评分量表
        medical_tools = []
        for match in scale_matches:
            tool_name = match.strip()
            # 查找包含医疗评分关键词的工具
            if any(keyword in tool_name.lower() for keyword in [
                'score', 'scale', 'assessment', 'index', 'calculator',
                'apache', 'sofa', 'curb', 'centor', 'glasgow', 'heart'
            ]):
                medical_tools.append(tool_name)
                logger.debug(f"找到医疗工具: {tool_name}")
        
        if medical_tools:
            # 取最后一个医疗工具（通常是最终选择）
            final_tool = medical_tools[-1]
            logger.debug(f"最终医疗工具: {final_tool}")
            return final_tool
        
        # 备用方案1：从INFO日志中查找工具名称
        info_lines = log_content.split('\n')
        for line in reversed(info_lines):  # 从后往前找
            if 'INFO - ' in line and any(keyword in line.lower() for keyword in [
                'apache', 'sofa', 'curb', 'centor', 'glasgow', 'heart', 'score'
            ]):
                # 提取INFO后面的内容
                info_match = re.search(r'INFO - (.+)$', line)
                if info_match:
                    tool_name = info_match.group(1).strip()
                    # 过滤掉明显不是工具名的行
                    if not any(skip in tool_name.lower() for skip in [
                        'http', 'retrieve', 'calculate', 'extract', 'reflect',
                        'now in', 'cost time', 'chosen', 'json'
                    ]):
                        logger.debug(f"从INFO日志提取工具名称: {tool_name}")
                        return tool_name
        
        # 备用方案2：查找所有chosen_tool_name，取最后一个
        if scale_matches:
            final_tool = scale_matches[-1].strip()
            logger.debug(f"取最后一个chosen_tool_name: {final_tool}")
            return final_tool
        
        logger.warning("未能从日志中提取工具名称")
        return None
    
    def extract_score_from_log(self, log_content: str) -> Optional[float]:
        """
        从日志内容中提取计算分数
        
        Args:
            log_content: 日志文件内容
            
        Returns:
            提取到的分数，如果未找到则返回None
        """
        # 按行分割日志内容
        lines = log_content.strip().split('\n')
        
        # 从最后一行开始查找分数
        for line in reversed(lines):
            # 查找最后一行的 "INFO - 数字" 格式
            match = re.search(r'INFO - (\d+(?:\.\d+)?)$', line.strip())
            if match:
                try:
                    score = float(match.group(1))
                    logger.debug(f"从最后一行提取到分数: {score}")
                    return score
                except ValueError:
                    continue
        
        logger.warning("未能从日志中提取分数（可能未计算成功）")
        return None
    
    def normalize_tool_name(self, tool_name: str) -> str:
        """
        标准化工具名称以便比较
        
        Args:
            tool_name: 原始工具名称
            
        Returns:
            标准化后的工具名称
        """
        if not tool_name:
            return ""
        
        # 转换为小写
        normalized = tool_name.lower()
        
        # 移除常见的变体和标点
        normalized = re.sub(r'[^\w\s]', ' ', normalized)  # 移除标点
        normalized = re.sub(r'\s+', ' ', normalized)      # 多个空格合并为一个
        normalized = normalized.strip()
        
        # 处理常见的同义词
        synonyms = {
            'apache ii': 'apache ii score',
            'apache 2': 'apache ii score',
            'glasgow blatchford': 'glasgow blatchford bleeding score',
            'centor': 'centor score',
            'sofa': 'sofa score',
            'sequential organ failure assessment': 'sofa score',
        }
        
        for key, value in synonyms.items():
            if key in normalized:
                normalized = value
                break
        
        return normalized
    
    def compare_tool_names(self, standard_name: str, extracted_name: str) -> bool:
        """
        比较工具名称是否匹配
        
        Args:
            standard_name: 标准答案中的工具名称
            extracted_name: 从日志中提取的工具名称
            
        Returns:
            是否匹配
        """
        if not standard_name or not extracted_name:
            return False
        
        std_normalized = self.normalize_tool_name(standard_name)
        ext_normalized = self.normalize_tool_name(extracted_name)
        
        # 精确匹配
        if std_normalized == ext_normalized:
            return True
        
        # 部分匹配（包含关系）
        if std_normalized in ext_normalized or ext_normalized in std_normalized:
            return True
        
        # 关键词匹配
        std_keywords = set(std_normalized.split())
        ext_keywords = set(ext_normalized.split())
        
        # 如果有超过50%的关键词重叠，认为匹配
        if len(std_keywords & ext_keywords) / max(len(std_keywords), len(ext_keywords)) > 0.5:
            return True
        
        return False
    
    def compare_scores(self, standard_score: float, extracted_score: float, 
                      tolerance: float = 0.1) -> bool:
        """
        比较分数是否匹配（在容忍范围内）
        
        Args:
            standard_score: 标准答案分数
            extracted_score: 提取的分数
            tolerance: 容忍度（相对误差）
            
        Returns:
            是否匹配
        """
        if standard_score is None or extracted_score is None:
            return False
        
        # 精确匹配
        if standard_score == extracted_score:
            return True
        
        # 相对误差匹配
        if standard_score != 0:
            relative_error = abs(standard_score - extracted_score) / abs(standard_score)
            return relative_error <= tolerance
        
        # 标准分数为0时，使用绝对误差
        return abs(extracted_score) <= 1.0
    
    def evaluate_single_case(self, index: int) -> Dict:
        """
        评估单个案例
        
        Args:
            index: 案例索引
            
        Returns:
            评估结果字典
        """
        result = {
            'index': index,
            'tool_correct': False,
            'score_correct': False,
            'standard_tool': '',
            'extracted_tool': '',
            'standard_score': None,
            'extracted_score': None,
            'log_file': '',
            'error': ''
        }
        
        # 获取标准答案
        if index not in self.standard_data:
            result['error'] = f"标准答案中未找到索引 {index}"
            return result
        
        standard = self.standard_data[index]
        result['standard_tool'] = standard['calculator_name']
        result['standard_score'] = standard['calculator_score']
        
        # 查找对应的日志文件
        log_file = os.path.join(self.log_dir, f"3.5_{index}.log")
        result['log_file'] = log_file
        
        if not os.path.exists(log_file):
            result['error'] = f"日志文件不存在: {log_file}"
            self.results['missing_logs'].append(index)
            return result
        
        try:
            # 读取日志文件
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 提取工具名称和分数
            extracted_tool = self.extract_tool_name_from_log(log_content)
            extracted_score = self.extract_score_from_log(log_content)
            
            result['extracted_tool'] = extracted_tool or ''
            result['extracted_score'] = extracted_score
            
            # 比较工具名称
            if extracted_tool:
                result['tool_correct'] = self.compare_tool_names(
                    standard['calculator_name'], extracted_tool
                )
            
            # 比较分数
            if extracted_score is not None and standard['calculator_score'] is not None:
                result['score_correct'] = self.compare_scores(
                    standard['calculator_score'], extracted_score
                )
                logger.debug(f"分数比较: 标准={standard['calculator_score']}, 提取={extracted_score}, 匹配={result['score_correct']}")
            
        except Exception as e:
            result['error'] = f"处理日志文件时出错: {e}"
            self.results['error_logs'].append(index)
        
        return result
    
    def evaluate_all(self) -> Dict:
        """
        评估所有案例
        
        Returns:
            完整的评估结果
        """
        logger.info("开始评估所有案例...")
        
        # 加载标准数据
        self.load_standard_data()
        
        # 统计变量
        total_cases = 0
        tool_correct = 0
        score_correct = 0
        detailed_results = []
        tool_error_indices = []
        score_error_indices = []
        
        # 按索引顺序评估
        indices = sorted(self.standard_data.keys())
        
        for index in indices:
            logger.info(f"正在评估案例 {index}...")
            result = self.evaluate_single_case(index)
            detailed_results.append(result)
            
            total_cases += 1
            if result['tool_correct']:
                tool_correct += 1
            else:
                tool_error_indices.append(index)
                
            if result['score_correct']:
                score_correct += 1
            else:
                score_error_indices.append(index)
            
            # 输出单个案例的结果
            logger.info(f"案例 {index}: 工具{'✓' if result['tool_correct'] else '✗'} "
                       f"分数{'✓' if result['score_correct'] else '✗'}")
            
            if result['error']:
                logger.warning(f"案例 {index} 出现错误: {result['error']}")
        
        # 计算准确率
        tool_accuracy = (tool_correct / total_cases) * 100 if total_cases > 0 else 0
        score_accuracy = (score_correct / total_cases) * 100 if total_cases > 0 else 0
        
        # 更新结果
        self.results.update({
            'tool_accuracy': tool_accuracy,
            'score_accuracy': score_accuracy,
            'total_cases': total_cases,
            'tool_correct': tool_correct,
            'score_correct': score_correct,
            'detailed_results': detailed_results,
            'tool_error_indices': tool_error_indices,
            'score_error_indices': score_error_indices
        })
        
        return self.results
    
    def generate_txt_report(self) -> str:
        """
        生成详细的评估报告
        
        Args:
            output_file: 输出文件路径
        """
        logger.info("生成评估报告...")
        
        # 创建报告
        report = {
            'summary': {
                'total_cases': self.results['total_cases'],
                'tool_accuracy': f"{self.results['tool_accuracy']:.2f}%",
                'score_accuracy': f"{self.results['score_accuracy']:.2f}%",
                'tool_correct_count': self.results['tool_correct'],
                'score_correct_count': self.results['score_correct'],
                'missing_logs_count': len(self.results['missing_logs']),
                'error_logs_count': len(self.results['error_logs'])
            },
            'detailed_results': self.results['detailed_results'],
            'missing_logs': self.results['missing_logs'],
            'error_logs': self.results['error_logs']
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存到: {output_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("MeNTi 工具选择和分数正确率评估报告")
        print("="*60)
        print(f"总案例数: {report['summary']['total_cases']}")
        print(f"工具选择正确率: {report['summary']['tool_accuracy']}")
        print(f"分数计算正确率: {report['summary']['score_accuracy']}")
        print(f"工具选择正确数: {report['summary']['tool_correct_count']}")
        print(f"分数计算正确数: {report['summary']['score_correct_count']}")
        print(f"缺失日志文件数: {report['summary']['missing_logs_count']}")
        print(f"错误日志文件数: {report['summary']['error_logs_count']}")
        
        if self.results['missing_logs']:
            print(f"\n缺失的日志文件对应的索引: {self.results['missing_logs'][:10]}...")
        
        if self.results['error_logs']:
            print(f"处理出错的日志文件对应的索引: {self.results['error_logs'][:10]}...")
        
        print("="*60)


def test_single_case():
    """测试单个案例"""
    evaluator = MentiEvaluator()
    evaluator.load_standard_data()
    
    # 测试index=3的案例（有完整计算结果）
    result = evaluator.evaluate_single_case(3)
    
    print("=" * 50)
    print("单案例测试结果:")
    print("=" * 50)
    print(f"案例索引: {result['index']}")
    print(f"标准工具: {result['standard_tool']}")
    print(f"提取工具: {result['extracted_tool']}")
    print(f"工具匹配: {'✓' if result['tool_correct'] else '✗'}")
    print(f"标准分数: {result['standard_score']}")
    print(f"提取分数: {result['extracted_score']}")
    print(f"分数匹配: {'✓' if result['score_correct'] else '✗'}")
    print(f"日志文件: {result['log_file']}")
    if result['error']:
        print(f"错误信息: {result['error']}")
    print("=" * 50)
    
    return result


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 测试模式
        return test_single_case()
    else:
        # 完整评估模式
        evaluator = MentiEvaluator()
        results = evaluator.evaluate_all()
        evaluator.generate_report("menti_evaluation_report.json")
        return results


if __name__ == "__main__":
    main()
