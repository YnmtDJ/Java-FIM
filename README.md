# Java代码补全
## 任务介绍
随着大模型技术的飞速发展，大模型在帮助程序员完成编码任务上取得了巨大提升。根据当前代码的上下文自动生成代码片段（FIM，Fill In the Middle）就是其中的一项主要任务。它能够节省研发人员编程成本，提高编码效率。  
代码补全示例：  
```json
{
  "prompt": "<｜fim▁begin｜>def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = []\n    right = []\n<｜fim▁hole｜>        if arr[i] < pivot:\n            left.append(arr[i])\n        else:\n            right.append(arr[i])\n    return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>",
  "response": "    for i in range(1, len(arr)):\n"
}
```

## DataTool
数据集生成工具，主要用于生成代码补全数据集。首先需要在[Java-GitHub-Codes](https://huggingface.co/datasets/hrishizone/Java-GitHub-Codes)上下载Java代码数据集。然后，运行`java_github_codes.py`脚本来构造prompt和response微调数据集。  
在`java_github_codes.py`主函数开头可设置相关参数：  
```text
data_files - Java-GitHub-Codes数据集路径
train_save_path - 训练集保存路径
eval_save_path - 验证集保存路径
num_samples - 数据集采样数量
num_samples_per_code - 每个Java代码文件采样数量
train_eval_split - 训练集和验证集划分比例
max_length - 上下文最大长度
min_lines - 代码最少行数
max_hole_lines - 最大挖取行数
```

## EvaTool
EvaTool参考官方给定的[评估工具](https://uploadfiles.nowcoder.com/files/20250508/304226_1746691654917/%E8%AF%84%E6%B5%8B%E5%B7%A5%E5%85%B7%E5%8F%8A%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8Fv2.zip)，而编写的评估大模型生成结果的代码。首先需要通过LLaMA-Factory工具生成大模型在验证集上的预测结果`generated_predictions.jsonl`。然后，运行`main.py`脚本，将LLaMA-Factory生成的预测结果转换为指定的评估格式`generated.jsonl`。同时，计算[Exact Match](https://huggingface.co/spaces/evaluate-metric/exact_match/blame/72aadca217d0152597936b0d1c16485476a83ce7/exact_match.py)评估指标，保存到`output.csv`文件中。  
在`main.py`主函数开头可设置相关参数：
```text
predict_path - LLaMA-Factory生成的预测结果路径
output_path - 格式转换后的输出文件路径
result_path - 评估结果保存路径
```

## LLaMA-Factory
LLaMA-Factory官网源代码。为了添加自定义数据集和选择[DeepSeek-Coder-1.3B](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)进行微调，需要修改两处代码：  
- `data/dataset_info.json`文件中，添加自定义数据集信息，并把数据集加入`data/`目录下：  
```json
{
  "java_fim": {
    "file_name": "java_fim.json",
    "columns": {
      "prompt": "prompt",
      "response": "response"
    }
  },
  "java_fim_eval": {
    "file_name": "java_fim_eval.json",
    "columns": {
      "prompt": "prompt",
      "response": "response"
    }
  }
}
```
- `src/llamafactory/extras/constants.py`文件中，注册模型相关信息：
```python
register_model_group(
    models={
        "DeepSeek-Coder-1.3B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-1.3b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-1.3b-base",
        },
        ...
    }
    template="deepseekcoder",
)
```

## script
- `train_shell`：llamafactory训练脚本
```shell
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /home/cti/project/python/model/deepseek-ai/deepseek-coder-1___3b-base \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template empty \
    --flash_attn auto \
    --dataset_dir data \
    --dataset java_fim \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/DeepSeek-Coder-1.3B-Base/lora/train \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```
- `predict_shell`：llamafactory预测脚本
```shell
llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/cti/project/python/model/deepseek-ai/deepseek-coder-1___3b-base \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template empty \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset java_fim_eval \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 256 \
    --top_p 1 \
    --temperature 0.1 \
    --output_dir saves/DeepSeek-Coder-1.3B-Base/lora/eval \
    --trust_remote_code True \
    --do_predict True \
    --adapter_name_or_path saves/DeepSeek-Coder-1.3B-Base/lora/train
```