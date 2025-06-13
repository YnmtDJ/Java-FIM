import csv
import json

from EvalTool.data_process import evaluation


def convert_predict_format(predict_path: str, output_path: str):
    """
    将llama-factory生成的预测结果转换为指定的评估格式
    """
    output_lines = []
    with open(predict_path, "r", encoding="utf-8") as f:
        for line in f:
            json_obj = json.loads(line)
            json_obj.pop('prompt')
            json_obj['generated_code'] = json_obj.pop('predict')
            json_obj['reference_code'] = json_obj.pop('label')
            json_obj['case_type'] = 'block'
            output_lines.append(json.dumps(json_obj, ensure_ascii=False))

    with open(output_path, 'w', encoding="utf-8") as file:
        file.write('\n'.join(output_lines))



if __name__ == "__main__":
    predict_path = "./generated_predictions.jsonl"  # llama-factory生成的预测结果文件
    output_path = "./generated.jsonl"  # 转换后的输出文件
    result_path = "./output.csv"  # 评估结果输出文件

    convert_predict_format(predict_path, output_path)

    em_scores = evaluation.evaluation(output_path)
    with open(result_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(em_scores.keys())  # 写入表头（键）
        writer.writerow(em_scores.values())  # 写入数据（值）
