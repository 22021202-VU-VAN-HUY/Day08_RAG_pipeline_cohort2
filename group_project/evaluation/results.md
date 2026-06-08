# RAG Evaluation Results

**Date:** 2026-06-08 15:22

**Golden Dataset:** 15 Q&A pairs

**Evaluation Method:** Keyword-based metrics (no LLM evaluator)

---

## Config A: Hybrid + Rerank

| Metric | Score |
|--------|-------|
| faithfulness | 0.721 |
| answer_relevancy | 0.892 |
| context_recall | 0.656 |
| context_precision | 0.795 |

## Config B: Dense-Only (no rerank)

| Metric | Score |
|--------|-------|
| faithfulness | 0.663 |
| answer_relevancy | 0.864 |
| context_recall | 0.541 |
| context_precision | 0.703 |

## A/B Comparison

| Metric | Config A | Config B | Winner |
|--------|----------|----------|--------|
| faithfulness | 0.721 | 0.663 | **A** |
| answer_relevancy | 0.892 | 0.864 | **A** |
| context_recall | 0.656 | 0.541 | **A** |
| context_precision | 0.795 | 0.703 | **A** |

## Per-Question Detail (Config A: Hybrid + Rerank)

| # | Question | Faith. | Rel. | Recall | Prec. | Sources |
|---|----------|--------|------|--------|-------|--------|
| 1 | Hình phạt cho tội tàng trữ trái phép chấ... | 0.53 | 1.00 | 0.50 | 0.71 | 5 |
| 2 | Luật Phòng chống ma tuý 2021 quy định nh... | 0.49 | 0.93 | 0.43 | 0.64 | 5 |
| 3 | Danh mục các chất ma tuý thuộc nhóm I th... | 0.87 | 0.84 | 0.64 | 0.74 | 5 |
| 4 | Ca sĩ Chu Bin bị bắt vì lý do gì? | 0.73 | 0.80 | 0.65 | 0.70 | 5 |
| 5 | Châu Việt Cường bị xử lý như thế nào tro... | 0.64 | 0.88 | 0.59 | 0.88 | 5 |
| 6 | Nguyễn Đỗ Trúc Phương bị bắt vì liên qua... | 0.92 | 0.77 | 0.94 | 0.77 | 5 |
| 7 | Vụ 4 tiếp viên hàng không xách ma tuý li... | 0.81 | 0.93 | 0.79 | 0.86 | 5 |
| 8 | Nghị định 105/2021/NĐ-CP hướng dẫn những... | 0.76 | 0.72 | 0.72 | 0.83 | 5 |
| 9 | Tội vận chuyển trái phép chất ma tuý bị ... | 0.62 | 0.79 | 0.54 | 0.79 | 5 |
| 10 | Luật Phòng chống ma tuý 2021 quy định gì... | 0.81 | 0.80 | 0.55 | 0.87 | 5 |
| 11 | Các chất tiền chất ma tuý là gì và được ... | 0.67 | 0.93 | 0.81 | 0.64 | 5 |
| 12 | Đường dây ma tuý lớn nhất lịch sử Việt N... | 0.79 | 1.00 | 0.79 | 0.94 | 5 |
| 13 | Hình phạt cho tội sản xuất trái phép chấ... | 0.74 | 1.00 | 0.54 | 0.85 | 5 |
| 14 | Nghệ sĩ nào bị bắt vì liên quan đến ma t... | 0.75 | 1.00 | 0.86 | 0.85 | 5 |
| 15 | Bộ luật Hình sự 2015 Chương XX quy định ... | 0.70 | 1.00 | 0.50 | 0.88 | 5 |

## Worst Performers

- **Q:** Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào?
  - Scores: {'faithfulness': 0.487, 'answer_relevancy': 0.929, 'context_recall': 0.429, 'context_precision': 0.643}
  - Expected: Cai nghiện tự nguyện tại gia đình, cai nghiện tự nguyện tại cộng đồng, cai nghiện tự nguyện tại cơ s...
  - Got: Dựa trên các tài liệu được cung cấp, tôi không thể xác định chi tiết các hình thức cai nghiện được q...

- **Q:** Tội vận chuyển trái phép chất ma tuý bị phạt tù bao nhiêu năm?
  - Scores: {'faithfulness': 0.623, 'answer_relevancy': 0.786, 'context_recall': 0.538, 'context_precision': 0.786}
  - Expected: Tùy theo khối lượng ma tuý, tội vận chuyển trái phép chất ma tuý có thể bị phạt từ 2 năm đến tử hình...
  - Got: Theo các tài liệu được cung cấp, tôi không thể xác minh thông tin về mức hình phạt tù cụ thể cho tội...

- **Q:** Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự?
  - Scores: {'faithfulness': 0.531, 'answer_relevancy': 1.0, 'context_recall': 0.5, 'context_precision': 0.706}
  - Expected: Phạt tù từ 1 đến 5 năm đối với trường hợp tàng trữ từ 1g đến dưới 5g heroin hoặc cocaine, hoặc chất ...
  - Got: Dựa trên các tài liệu được cung cấp, tôi không thể tìm thấy thông tin về mức hình phạt cụ thể cho tộ...

## Recommendations

1. **Improve Vietnamese embedding**: Switch to BAAI/bge-m3 for better context recall
2. **Increase chunk overlap**: Current 50 chars may split important context
3. **PageIndex fallback**: Enable for complex reasoning queries
4. **Prompt engineering**: Refine system prompt for better citation format
