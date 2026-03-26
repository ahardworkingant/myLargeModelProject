import re
import math


def medical_calculator(query):
    """用于计算医学相关指标，如BMI、体表面积、肾小球滤过率(eGFR)等。输入应为需要计算的指标名称和数值。"""
    try:
        # 统一转换为小写以便匹配
        query_lower = query.lower()

        # BMI计算
        if "bmi" in query_lower or "体重指数" in query_lower:
            return calculate_bmi(query_lower)

        # 体表面积计算
        elif "体表面积" in query_lower or "bsa" in query_lower:
            return calculate_bsa(query_lower)

        # 肾小球滤过率计算
        elif "egfr" in query_lower or "肾小球滤过率" in query_lower or "肾功能" in query_lower:
            return calculate_egfr(query_lower)

        # 理想体重计算
        elif "理想体重" in query_lower or "ibw" in query_lower:
            return calculate_ibw(query_lower)

        # 肌酐清除率计算
        elif "肌酐清除率" in query_lower or "ccr" in query_lower:
            return calculate_ccr(query_lower)

        # 如果没有匹配到任何计算类型，返回提示信息
        else:
            return "我可以计算以下医学指标：BMI、体表面积(BSA)、肾小球滤过率(eGFR)、理想体重(IBW)、肌酐清除率(CCR)。请明确指定要计算的指标和参数。"

    except Exception as e:
        return f"计算过程中出错：{str(e)}。请确保提供了正确的参数格式。"


def calculate_bmi(query):
    """计算身体质量指数(BMI)"""
    # 使用正则表达式提取身高和体重
    height_match = re.search(r'身高[:\s]*(\d+\.?\d*)\s*(cm|厘米|米|m)', query)
    weight_match = re.search(r'体重[:\s]*(\d+\.?\d*)\s*(kg|公斤|千克)', query)

    if not height_match or not weight_match:
        return "请提供身高和体重信息，例如：'计算BMI，身高170cm，体重65kg'"

    height = float(height_match.group(1))
    weight = float(weight_match.group(1))
    height_unit = height_match.group(2)

    # 如果身高单位是厘米，转换为米
    if height_unit in ["cm", "厘米"]:
        height = height / 100

    # 计算BMI
    bmi = weight / (height ** 2)

    # 判断BMI分类
    if bmi < 18.5:
        category = "体重过轻"
    elif 18.5 <= bmi < 24:
        category = "正常体重"
    elif 24 <= bmi < 28:
        category = "超重"
    else:
        category = "肥胖"

    return f"您的BMI指数为：{bmi:.2f}，属于'{category}'范围。正常BMI范围为18.5-24。"


def calculate_bsa(query):
    """计算体表面积(BSA) - 使用Mosteller公式"""
    # 使用正则表达式提取身高和体重
    height_match = re.search(r'身高[:\s]*(\d+\.?\d*)\s*(cm|厘米)', query)
    weight_match = re.search(r'体重[:\s]*(\d+\.?\d*)\s*(kg|公斤|千克)', query)

    if not height_match or not weight_match:
        return "请提供身高(cm)和体重(kg)信息，例如：'计算体表面积，身高170cm，体重65kg'"

    height = float(height_match.group(1))  # 厘米
    weight = float(weight_match.group(1))  # 千克

    # Mosteller公式: BSA (m²) = √([身高(cm) × 体重(kg)] / 3600)
    bsa = math.sqrt((height * weight) / 3600)

    return f"您的体表面积(BSA)为：{bsa:.4f} 平方米(Mosteller公式)。"


def calculate_egfr(query):
    """估算肾小球滤过率(eGFR) - 使用CKD-EPI公式简化版"""
    # 使用正则表达式提取年龄、性别和肌酐值
    age_match = re.search(r'年龄[:\s]*(\d+)\s*岁?', query)
    scr_match = re.search(r'肌酐[:\s]*(\d+\.?\d*)\s*(μmol/L|umol/l|μmol|umol|mg/dl)?', query)
    gender_match = re.search(r'(男|女|男性|女性)', query)

    if not age_match or not scr_match or not gender_match:
        return "请提供年龄、性别和血清肌酐值，例如：'计算eGFR，年龄45岁，男性，肌酐80μmol/L'"

    age = int(age_match.group(1))
    scr = float(scr_match.group(1))
    gender = gender_match.group(1)

    # 如果肌酐单位是mg/dl，转换为μmol/L (1 mg/dl = 88.4 μmol/L)
    scr_unit = scr_match.group(2) if scr_match.group(2) else ""
    if "mg/dl" in scr_unit.lower():
        scr = scr * 88.4

    # 简化版CKD-EPI公式 (实际临床应用中应使用完整公式)
    if "男" in gender:
        # 男性公式
        if scr <= 80:
            egfr = 141 * (scr / 80) ** -0.411 * 0.993 ** age
        else:
            egfr = 141 * (scr / 80) ** -1.209 * 0.993 ** age
    else:
        # 女性公式
        if scr <= 80:
            egfr = 144 * (scr / 80) ** -0.329 * 0.993 ** age
        else:
            egfr = 144 * (scr / 80) ** -1.209 * 0.993 ** age

    # 判断肾功能分期
    if egfr >= 90:
        stage = "1期：肾功能正常"
    elif 60 <= egfr < 90:
        stage = "2期：轻度肾功能下降"
    elif 30 <= egfr < 60:
        stage = "3期：中度肾功能下降"
    elif 15 <= egfr < 30:
        stage = "4期：重度肾功能下降"
    else:
        stage = "5期：肾衰竭"

    return f"估算的肾小球滤过率(eGFR)为：{egfr:.2f} mL/min/1.73m²，属于{stage}。注：此为简化估算，临床诊断请咨询专业医生。"


def calculate_ibw(query):
    """计算理想体重(IBW)"""
    # 使用正则表达式提取身高和性别
    height_match = re.search(r'身高[:\s]*(\d+\.?\d*)\s*(cm|厘米)', query)
    gender_match = re.search(r'(男|女|男性|女性)', query)

    if not height_match or not gender_match:
        return "请提供身高(cm)和性别信息，例如：'计算理想体重，身高170cm，男性'"

    height = float(height_match.group(1))  # 厘米
    gender = gender_match.group(1)

    # 计算理想体重 (Devine公式)
    if "男" in gender:
        ibw = 50 + 0.9 * (height - 152)
    else:
        ibw = 45.5 + 0.9 * (height - 152)

    return f"您的理想体重约为：{ibw:.1f} kg (Devine公式)。"


def calculate_ccr(query):
    """计算肌酐清除率(Cockcroft-Gault公式)"""
    # 使用正则表达式提取年龄、性别、体重和肌酐值
    age_match = re.search(r'年龄[:\s]*(\d+)\s*岁?', query)
    weight_match = re.search(r'体重[:\s]*(\d+\.?\d*)\s*(kg|公斤|千克)', query)
    scr_match = re.search(r'肌酐[:\s]*(\d+\.?\d*)\s*(μmol/L|umol/l|μmol|umol|mg/dl)?', query)
    gender_match = re.search(r'(男|女|男性|女性)', query)

    if not age_match or not weight_match or not scr_match or not gender_match:
        return "请提供年龄、体重、性别和血清肌酐值，例如：'计算肌酐清除率，年龄45岁，体重70kg，男性，肌酐80μmol/L'"

    age = int(age_match.group(1))
    weight = float(weight_match.group(1))
    scr = float(scr_match.group(1))
    gender = gender_match.group(1)

    # 如果肌酐单位是μmol/L，转换为mg/dl (1 mg/dl = 88.4 μmol/L)
    scr_unit = scr_match.group(2) if scr_match.group(2) else ""
    if "μmol" in scr_unit.lower() or "umol" in scr_unit.lower():
        scr = scr / 88.4

    # Cockcroft-Gault公式
    if "男" in gender:
        ccr = ((140 - age) * weight) / (72 * scr)
    else:
        ccr = ((140 - age) * weight) / (72 * scr) * 0.85

    return f"估算的肌酐清除率(Cockcroft-Gault公式)为：{ccr:.2f} mL/min。注：此为估算值，临床诊断请咨询专业医生。"


# 测试函数
if __name__ == "__main__":
    test_queries = [
        "计算BMI，身高170cm，体重65kg",
        "计算体表面积，身高175cm，体重70kg",
        "计算eGFR，年龄45岁，男性，肌酐80μmol/L",
        "计算理想体重，身高180cm，男性",
        "计算肌酐清除率，年龄50岁，体重75kg，男性，肌酐90μmol/L",
        "我不会计算这个"
    ]

    for query in test_queries:
        print(f"问题: {query}")
        print(f"回答: {medical_calculator(query)}")
        print("-" * 50)