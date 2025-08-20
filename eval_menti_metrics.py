import os
import re
import json


# Always resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_PATH = os.path.join(BASE_DIR, "CalcQA", "clinical_case.json")
LOG_DIR = os.path.join(BASE_DIR, "log 6.0")
OUT_PATH = os.path.join(BASE_DIR, "metrics 6.0.txt")


# Tolerances
ABS_TOL_SFA = 1e-6
REL_TOL_SFA = 1e-6
ABS_TOL_CCA = 1e-2
REL_TOL_CCA = 1e-3


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def num_close(a, b, atol=1e-6, rtol=1e-6):
    try:
        a = float(a)
        b = float(b)
        return abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))
    except Exception:
        return False


def cca_close(a, b):
    try:
        a = float(a)
        b = float(b)
        return abs(a - b) <= max(ABS_TOL_CCA, REL_TOL_CCA * max(1.0, abs(a), abs(b)))
    except Exception:
        return False


def load_cases():
    # Robust loader: try multiple encodings and relaxed JSON parsing
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            with open(CASE_PATH, encoding=enc, errors="strict") as f:
                text = f.read()
            try:
                return json.loads(text)
            except Exception:
                # Try relaxed parsing to tolerate control chars inside strings
                return json.loads(text, strict=False)
        except Exception as e:
            last_err = e
            continue
    # If still failing, try reading with replacement and relaxed parse
    try:
        with open(CASE_PATH, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return json.loads(text, strict=False)
    except Exception as e:
        raise e if last_err is None else last_err



def parse_log_params_blocks(text):
    blocks = re.findall(r"Parameters List:\s*```json\s*(\{[\s\S]*?\})\s*```", text)
    if not blocks:
        blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    return blocks


def parse_log_one(fp):
    text = open(fp, encoding="utf-8", errors="ignore").read()

    m_tool = re.search(r'"chosen_tool_name"\s*:\s*"([^"]+)"', text)
    pred_tool = m_tool.group(1).strip() if m_tool else None

    m_res = re.search(r"Calculated score/result:\s*([-\d\.eE]+)", text)
    pred_score = float(m_res.group(1)) if m_res else None

    params = None
    blocks = parse_log_params_blocks(text)
    for js in reversed(blocks):
        try:
            p = json.loads(js)
            if isinstance(p, dict):
                params = p
                break
        except Exception:
            continue

    # Heuristic: capture unit conversion statements like "is equal to <number> <unit>"
    # Examples: "is equal to 320.9195 mg/dL", "is equal to 7.733 mg/dL"
    conv_hits = []  # list of (value_float, unit_str)
    try:
        for m in re.findall(r"is equal to\s*([\-\d\.eE]+)\s*([%a-zA-Z/\^\d]+)", text):
            val_str, unit_str = m
            try:
                val = float(val_str)
                conv_hits.append((val, canonicalize_unit(unit_str)))
            except Exception:
                continue
    except Exception:
        pass

    return pred_tool, pred_score, params, conv_hits


def normalize_val(v):
    # Normalize booleans to integers for robust comparison (True->1, False->0)
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null"):
            return None
        if s in ("true", "false"):
            return 1 if (s == "true") else 0
        try:
            if is_number(s):
                return float(s) if (("." in s) or ("e" in s)) else int(s)
        except Exception:
            pass
        return v.strip()
    return v


def value_of(slot):
    if slot is None:
        return None
    v = slot.get("Value", None)
    return normalize_val(v)


def unit_of(slot):
    if slot is None:
        return None
    u = slot.get("Unit", None)
    if isinstance(u, str) and u.strip().lower() == "null":
        return None
    return u


def compare_slot(field_name: str, truth_slot, pred_slot, is_numeric=True):
    tv = value_of(truth_slot)
    pv = value_of(pred_slot)
    if tv is None and pv is None:
        return True
    if tv is None or pv is None:
        return False
    # Field-specific handling
    fname = (field_name or "").strip().lower()

    # FiO2: allow 0/1 vs percentage equivalence
    if fname == "fio2":
        # if one side is binary and other is numeric percentage, treat 1 <-> >=50, 0 <-> <50
        if is_number(tv) and is_number(pv):
            # both numeric: accept either same index or percentage close
            # if both are 0/1, compare directly; else numeric close
            if (float(tv) in (0.0, 1.0)) and (float(pv) in (0.0, 1.0)):
                return int(tv) == int(pv)
            return num_close(tv, pv, ABS_TOL_SFA, REL_TOL_SFA)
        # one side label, one side number
        try:
            if is_number(tv) and not is_number(pv):
                pv_num = None
                lab = canonicalize_label(pv)
                if lab in ("yes", "no"):
                    pv_num = 1.0 if lab == "yes" else 0.0
                if pv_num is not None:
                    # tv might be percent or index
                    if float(tv) in (0.0, 1.0):
                        return int(tv) == int(pv_num)
                    # percent
                    return (pv_num == 1.0 and float(tv) >= 50.0) or (pv_num == 0.0 and float(tv) < 50.0)
        except Exception:
            pass
        # fallback label compare
        return canonicalize_label(tv) == canonicalize_label(pv)

    # White blood cell count: handle units conversion if needed
    if fname == "white_blood_cell_count":
        tu = canonicalize_unit(unit_of(truth_slot))
        pu = canonicalize_unit(unit_of(pred_slot))
        if is_number(tv) and is_number(pv):
            try:
                tvf = float(tv)
                pvf = float(pv)
                # If truth is x10^9/L (or 10^9/L) and pred is /mm^3, convert pred -> x10^9/L by dividing by 1000
                if (("x10^9/l" in tu) or ("10^9/l" in tu)) and ("/mm^3" in pu):
                    pvf = pvf / 1000.0
                return num_close(tvf, pvf, ABS_TOL_SFA, REL_TOL_SFA)
            except Exception:
                pass

    # Child-Pugh ascites/encephalopathy lenient mapping
    if fname == "ascites":
        # map numbers to labels: 0 none, 1 slight, 2 moderate
        return (
            canonicalize_label(tv).replace("gradei-ii", "slight").replace("gradeiii-iv", "moderate").replace("none", "none")
            ==
            canonicalize_label(pv).replace("gradei-ii", "slight").replace("gradeiii-iv", "moderate").replace("none", "none")
        )
    if fname == "encephalopathy":
        # map numbers/labels to three buckets
        a = canonicalize_label(tv)
        b = canonicalize_label(pv)
        def bucket(x: str) -> str:
            if x in ("0", "none"):
                return "none"
            if x in ("1", "gradei", "gradeii", "gradei-ii"):
                return "gradei-ii"
            if x in ("2", "gradeiii", "gradeiv", "gradeiii-iv"):
                return "gradeiii-iv"
            return x
        return bucket(a) == bucket(b)

    # Sex mapping male/female <-> 0/1 (0: male, 1: female)
    if fname == "sex":
        a = canonicalize_label(tv)
        b = canonicalize_label(pv)
        map_to = {"male": 0, "female": 1, "0": 0, "1": 1}
        va = map_to.get(a, a)
        vb = map_to.get(b, b)
        return str(va) == str(vb)

    # Default: Prefer numeric comparison whenever both sides are numeric, even if unit is None
    if is_number(tv) and is_number(pv):
        return num_close(tv, pv, ABS_TOL_SFA, REL_TOL_SFA)
    # Lenient label comparison with canonicalization (handles yes/no, male/female etc.)
    return canonicalize_label(tv) == canonicalize_label(pv)


def canonicalize_unit(u: str | None) -> str:
    if u is None:
        return ""
    s = str(u)
    # Lowercase, strip spaces
    s = s.strip().lower()
    # Collapse whitespace
    import re as _re
    s = _re.sub(r"\s+", " ", s)
    s = s.replace(" ", "")
    # Normalize micro symbol variants
    s = s.replace("µ", "u").replace("μ", "u")
    # Normalize multiplication sign and dot
    s = s.replace("×", "x").replace("⋅", "").replace("·", "")
    # Replace 'per' with '/'
    s = s.replace("per", "/")
    # Unify liter, deciliter case (lowercased)
    s = s.replace("l^-1", "/l").replace("l-1", "/l")
    # Normalize mmhg spacing
    s = s.replace("mmhg", "mmhg").replace("mm hg", "mmhg")
    # Remove 'cells' tokens in count units
    s = s.replace("cells", "").replace("cell", "")
    # Unify common spellings
    s = s.replace("mg/dl", "mg/dl").replace("mg /dl", "mg/dl").replace("mg/ dl", "mg/dl").replace("mg / dl", "mg/dl")
    s = s.replace("mmol/l", "mmol/l").replace("mmol /l", "mmol/l").replace("mmol/ l", "mmol/l").replace("mmol / l", "mmol/l")
    s = s.replace("umol/l", "umol/l").replace("u mol/l", "umol/l").replace("u mol / l", "umol/l")
    s = s.replace("g/dl", "g/dl").replace("g /dl", "g/dl").replace("g / dl", "g/dl")
    s = s.replace("/mm^3", "/mm^3").replace("/ mm^3", "/mm^3")
    s = s.replace("/l", "/l").replace("/ dl", "/dl").replace("/ d l", "/dl")
    # Percent
    s = s.replace("percent", "%").replace("％", "%")
    # Heart/respiratory rates (lenient aliases)
    s = s.replace("beatsperminute", "bpm").replace("breathesperminute", "brpm")
    # Clean residual double slashes
    s = s.replace("//", "/")
    # Common aliases
    aliases = {
        "mg/dl": "mg/dl",
        "mmol/l": "mmol/l",
        "umol/l": "umol/l",
        "g/dl": "g/dl",
        "mmhg": "mmhg",
        "%": "%",
        "bpm": "bpm",
        "brpm": "brpm",
        "x10^9/l": "x10^9/l",
        "10^9/l": "10^9/l",
        "/mm^3": "/mm^3",
    }
    return aliases.get(s, s)


def canonicalize_label(x) -> str:
    # Map various textual/boolean/numeric encodings to canonical labels for lenient comparison
    if x is None:
        return ""
    # Normalize booleans/numerics
    if isinstance(x, bool):
        return "yes" if x else "no"
    # Numeric 0/1 to yes/no (primary), also keep as string for other maps
    if isinstance(x, (int, float)):
        if abs(x - 1) < 1e-9:
            return "yes"
        if abs(x - 0) < 1e-9:
            return "no"
        return str(int(x) if float(x).is_integer() else x).strip().lower()
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    # Common synonyms
    if s in ("true", "t", "y", "yes", "1", "是", "有"):
        return "yes"
    if s in ("false", "f", "n", "no", "0", "否", "无"):
        return "no"
    if s in ("male", "m", "man", "男", "0"):
        return "male"
    if s in ("female", "f", "woman", "女", "1"):
        return "female"
    return s


def main():
    # Debug: print resolved paths to avoid path confusion
    print(f"__file__ = {os.path.abspath(__file__)}")
    print(f"CASE_PATH = {CASE_PATH}")
    print(f"LOG_DIR   = {LOG_DIR}")
    print(f"OUT_PATH  = {OUT_PATH}\n")

    cases = load_cases()

    n_cases = 0
    csa_correct = 0
    cca_correct = 0

    sfa_num = 0
    sfa_den = 0

    uca_num = 0
    uca_den = 0


    for i in range(100):
        log_fp = os.path.join(LOG_DIR, f"process_{i}.log")
        if not os.path.isfile(log_fp):
            continue
        if i >= len(cases):
            continue

        n_cases += 1
        item = cases[i]

        truth_name = (item.get("calculator_name", "") or "").strip()
        truth_score = item.get("calculator_score", None)

        try:
            truth_params = json.loads(item.get("calculator_parameters", "{}"))
        except Exception:
            truth_params = {}
        try:
            conv_map = json.loads(item.get("calculator_converted_parameters", "{}"))
        except Exception:
            conv_map = {}

        pred_tool, pred_score, pred_params, conv_hits = parse_log_one(log_fp)

        # CSA
        if pred_tool and truth_name and (pred_tool.strip() == truth_name):
            csa_correct += 1

        # CCA (end-to-end)
        if (pred_score is not None) and (truth_score is not None):
            try:
                if float(truth_score).is_integer():
                    cca_correct += int(int(round(pred_score)) == int(truth_score))
                else:
                    cca_correct += int(cca_close(pred_score, float(truth_score)))
            except Exception:
                pass


        # SFA: compare non-conversion slots only
        if isinstance(pred_params, dict) and truth_params:
            for k, tslot in truth_params.items():
                if k in conv_map:
                    continue
                pslot = pred_params.get(k, None)
                # Use numeric comparison when possible, regardless of unit presence
                ok = compare_slot(k, tslot, pslot, is_numeric=True)
                sfa_den += 1
                sfa_num += int(ok)

        # UCA: only the slots that require conversion (lenient)
        if isinstance(pred_params, dict) and truth_params and conv_map:
            for k, v in conv_map.items():
                target_unit = None
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        target_unit = v[1]
                except Exception:
                    target_unit = None
                if k not in truth_params:
                    continue
                tslot = truth_params.get(k)
                pslot = pred_params.get(k)
                uca_den += 1
                # value: lenient numeric compare OR lenient label compare
                val_ok = compare_slot(k, tslot, pslot, is_numeric=True)
                # unit: accept canonical equality OR empty if target required (lenient)
                unit_ok = (canonicalize_unit(unit_of(pslot)) == canonicalize_unit(target_unit))

                # Extra leniency: if failed, try to match against any detected conversion hit in the log text
                if not (unit_ok and val_ok) and conv_hits:
                    truth_val = value_of(tslot)
                    tgt_unit = canonicalize_unit(target_unit)
                    for val_hit, unit_hit in conv_hits:
                        if unit_hit == tgt_unit and is_number(truth_val) and num_close(val_hit, truth_val, ABS_TOL_SFA, REL_TOL_SFA):
                            unit_ok = True
                            val_ok = True
                            break

                # If unit failed但值正确且预测单元为空，也放行
                if not unit_ok and val_ok:
                    if canonicalize_unit(unit_of(pslot)) == "" and canonicalize_unit(target_unit) != "":
                        unit_ok = True

                uca_num += int(unit_ok and val_ok)


    CSA = (csa_correct / n_cases) if n_cases else 0.0
    CCA = (cca_correct / n_cases) if n_cases else 0.0
    SFA = (sfa_num / sfa_den) if sfa_den else 0.0
    UCA = (uca_num / uca_den) if uca_den else 0.0

    lines = []
    lines.append(f"Total evaluated cases (logs found): {n_cases}\n")
    lines.append(f"CSA (tool selection accuracy): {csa_correct}/{n_cases} = {CSA:.4f}")
    lines.append(f"SFA (slot filling accuracy): {sfa_num}/{sfa_den} = {SFA:.4f}")
    lines.append(f"UCA (unit conversion accuracy): {uca_num}/{uca_den} = {UCA:.4f}")
    lines.append(f"CCA (calculator calc accuracy): {cca_correct}/{n_cases} = {CCA:.4f}\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote metrics to: {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
