for i in $(seq 0 100); do
    python MENTI.py \
        --test \
        --llm_model gpt-4 \
        --embedding_model m3e \
        --eval_index $i \
        --case_path "./CalcQA/clinical_case.json" \
        --tool_scale_path "./CalcQA/tool_scale.json" \
        --tool_unit_path "./CalcQA/tool_unit.json"
done