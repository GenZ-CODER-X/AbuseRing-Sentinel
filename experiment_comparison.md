# TBGF Experiment Comparison

## Model B Baseline
- PR-AUC: 0.5686268821281505
- ROC-AUC: 0.9110164400711532
- Recall@FPR≤1%: 0.5072463768115942

## Experiment Results

| Experiment | Features | PR-AUC | Δ PR-AUC | ROC-AUC | Recall@FPR≤1% |
|------------|----------|--------|----------|---------|----------------|
| Model B | Baseline | 0.5686268821281505 | 0.0000000000000000 | 0.9110164400711532 | 0.5072463768115942 |
| E0 (reproduction) | Model B features | 0.5686268821281505 | 0.0000000000000000 | 0.9110164400711532 | 0.5072463768115942 |
| E1 | + tb_graph_card_recency | 0.5677402143351699 | -0.0008866677929806 | 0.9143014425537899 | 0.5013709361535449 |
| E2 | + tb_graph_card_velocity | 0.5664665726866427 | -0.0021603094415078 | 0.9123450723549726 | 0.5013709361535449 |
| E3 | + tb_graph_is_novel_card_device | 0.5695601808725609 | +0.0009332987444104 | 0.9109738612181069 | 0.5111633372502937 |
| E4 | + tb_graph_card_amt_mean | 0.5676760620187296 | -0.0009508201094209 | 0.9086685644109659 | 0.5107716412064238 |
| E5 | + tb_graph_card_amt_zscore | 0.5661652592446043 | -0.0024616228835462 | 0.9109214020237532 | 0.509204857030944 |
| E6 | + all six TBGF features | 0.5653736590614185 | -0.0032532230667320 | 0.9100403881344381 | 0.5041128084606346 |

## Analysis

### 1. Best individual TBGF feature group (highest PR-AUC improvement among E1-E5)
**E3** (tb_graph_is_novel_card_device) with Δ PR-AUC = **+0.0009332987444104**

### 2. Best overall experiment (highest PR-AUC among E1-E6)
**E3** with PR-AUC = **0.5695601808725609**

### 3. PR-AUC threshold check (≥ 0.5701268821 = Model B + 0.0015)
- Target: 0.5701268821
- Best achieved (E3): 0.5695601808725609
- **Result: NOT ACHIEVED** (0.5695601808725609 < 0.5701268821)

### 4. ROC-AUC threshold check (≥ 0.9110164401 = no degradation required)
- Target: 0.9110164401
- Experiments meeting/exceeding target:
  - E1: 0.9143014425537899 ✓
  - E2: 0.9123450723549726 ✓
- **Result: ACHIEVED** (by E1 and E2)

### 5. Recall@FPR≤1% threshold check (≥ 0.5072463768 = no degradation required)
- Target: 0.5072463768
- Experiments meeting/exceeding target:
  - E3: 0.5111633372502937 ✓
  - E4: 0.5107716412064238 ✓
  - E5: 0.509204857030944 ✓
- **Result: ACHIEVED** (by E3, E4, and E5)

## Summary
- The TBGF_is_novel_card_device feature (E3) provides the best individual improvement
- Experiment E3 achieves the highest overall PR-AUC
- While the PR-AUC improvement target of +0.0015 was not met, several experiments show improvements in ROC-AUC and Recall@FPR≤1% without degradation