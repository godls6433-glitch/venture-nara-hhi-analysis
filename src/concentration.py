"""업체별 금액 집중도(HHI·CR) 계산과 추세검정 함수."""
import numpy as np
import pandas as pd
from scipy import stats

FIRM = "벤처업체번호"   # 업체 식별 키 (업체명은 표기 변형이 있어 번호 사용)
AMT = "실적금액"


def concentration(g: pd.DataFrame) -> pd.Series:
    """한 그룹(연도, 산업군×연도 등)의 참여업체 수와 금액 집중도 지표를 계산한다."""
    firm_amt = g.groupby(FIRM)[AMT].sum().sort_values(ascending=False)
    share = firm_amt / firm_amt.sum() * 100        # 업체별 점유율(%)
    hhi = (share ** 2).sum()                       # HHI (0~10,000)
    n = len(firm_amt)
    return pd.Series({
        "참여업체수": n,
        "거래건수": len(g),
        "총실적금액(억원)": firm_amt.sum() / 1e8,
        "업체당평균(백만원)": firm_amt.mean() / 1e6,
        "업체당중앙값(백만원)": firm_amt.median() / 1e6,
        "HHI": hhi,
        "정규화HHI": (hhi / 1e4 - 1 / n) / (1 - 1 / n) if n > 1 else 1.0,  # 업체수 효과 제거(0~1)
        "유효업체수(1/HHI)": 1e4 / hhi,
        "CR1(%)": share.iloc[:1].sum(),
        "CR3(%)": share.iloc[:3].sum(),
        "CR10(%)": share.iloc[:10].sum(),
        "상위10%업체점유(%)": share.iloc[:max(1, int(np.ceil(n * 0.1)))].sum(),
    })


def hhi_grade(h: float) -> str:
    """미국 DOJ/FTC 가이드라인의 3단계 집중도 등급."""
    if h < 1000:
        return "비집중(<1000)"
    return "중간집중(1000~1800)" if h <= 1800 else "고집중(>1800)"


def trend_label(rho: float, p_s: float, p_l: float) -> str:
    """Spearman·선형회귀 p값으로 추세 결론을 붙인다."""
    d = "증가" if rho > 0 else "감소"
    if p_s < 0.05 and p_l < 0.05:
        return f"유의한 {d}"            # 두 방법 모두 p<0.05
    if min(p_s, p_l) < 0.05 or max(p_s, p_l) < 0.10:
        return f"{d} 경향"              # 한 방법만 유의하거나 둘 다 경계(0.05~0.10)
    return "뚜렷한 추세 없음"


def trend_table(data: pd.DataFrame,
                cols=("참여업체수", "HHI", "정규화HHI", "CR3(%)", "유효업체수(1/HHI)")) -> pd.DataFrame:
    """연도 인덱스 데이터의 지표별 추세검정 결과표."""
    rows = []
    for col in cols:
        rho, p_s = stats.spearmanr(data.index, data[col])
        lr = stats.linregress(data.index, data[col])
        rows.append({"지표": col, "Spearman ρ": round(rho, 3), "Spearman p": round(p_s, 4),
                     "기울기(/년)": round(lr.slope, 2), "선형 p": round(lr.pvalue, 4),
                     "결론": trend_label(rho, p_s, lr.pvalue)})
    return pd.DataFrame(rows).set_index("지표")
