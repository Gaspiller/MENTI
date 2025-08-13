import os, json, re, sys, argparse
from pathlib import Path

# ------- 1) 读 prompt -------
BASE_DIR = Path(__file__).parent.resolve()
PROMPT_PATH = BASE_DIR / "extract_prediction.prompt"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT_TMPL = f.read()

# ------- 2) OpenAI 客户端（可换供应商）-------
from openai import OpenAI
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

# ------- 3) JSON 校验（防守）-------
def validate_schema(obj, schema_path=BASE_DIR / "prediction_schema.json"):
    try:
        import jsonschema
    except Exception:
        raise RuntimeError("pip install jsonschema")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=obj, schema=schema)

# ------- 4) 调 LLM 抽取 -------
def extract_one_from_log(
    raw_log: str,
    model: str = "gpt-3.5-turbo",
    prompt_template: str | None = None,
    schema_path: Path | None = None,
) -> dict:
    prompt_body = (prompt_template if prompt_template is not None else PROMPT_TMPL)
    prompt = prompt_body.replace("{{RAW_LOG}}", raw_log)
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":prompt}],
    )
    # OpenAI SDK 新版可直接拿 parsed；老版拿 content 后 json.loads
    data = resp.choices[0].message
    parsed = getattr(data, "parsed", None)
    obj = parsed if parsed is not None else json.loads(data.content)
    # schema 校验
    validate_schema(obj, schema_path=schema_path or (BASE_DIR / "prediction_schema.json"))
    return obj

# ------- 5) 从文件名猜 index（可选）-------
def try_fill_index_from_filename(obj: dict, filename: str):
    if obj.get("index") is not None:
        return obj
    m = re.search(r"(\d+)", filename)
    if m:
        obj["index"] = int(m.group(1))
    return obj

# ------- 6) CLI -------
def main():
    parser = argparse.ArgumentParser(description="Extract a structured prediction JSON from a raw log using LLM")
    parser.add_argument("--log_path", required=True, help="Path to raw log file")
    parser.add_argument("--out_path", required=True, help="Path to save extracted JSON")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model id, default gpt-3.5-turbo")
    parser.add_argument("--schema", default=str(BASE_DIR / "prediction_schema.json"), help="Path to JSON schema file")
    parser.add_argument("--prompt", default=str(PROMPT_PATH), help="Path to prompt template file")
    parser.add_argument("--fill_index_from_filename", action="store_true", help="If index is missing, infer from filename digits")
    args = parser.parse_args()

    raw = Path(args.log_path).read_text(encoding="utf-8", errors="ignore")
    prompt_tmpl = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else PROMPT_TMPL
    obj = extract_one_from_log(raw, model=args.model, prompt_template=prompt_tmpl, schema_path=Path(args.schema))
    if args.fill_index_from_filename:
        obj = try_fill_index_from_filename(obj, os.path.basename(args.log_path))
    Path(args.out_path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_path}\n", json.dumps(obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
