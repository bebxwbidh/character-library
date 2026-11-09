# character-library 人物资产库

婚纱套图 skill 的人物素材库，与 `scene-library`（场景库）配套。

- `index.yaml` — 24 条索引：性别/景别/发型/妆造/光源/同人组/人脸像素/锚点等级
- `characters/<id>/char.jpg` — 2048px 压缩版（生图参考用）
- `thumbs/<id>.jpg` — 480px 缩略图（目检挑人用）
- `matching-rules.md` — 使用规则（来源裁决、锚点等级、同人组、男女配对）
- `build_index.py` — 建库脚本（压缩 + facexlib 人脸检测 + 索引生成）

锚点等级：A=脸≥600px 可直接当锚点（22 张）｜B=300~599 建议先修复（2 张）｜C=<300 只作参考。

更新：图片放入源目录（如 `D:/人物资产`）→ 在 `build_index.py` 的 LABELS 补语义标注 → 重跑 `python build_index.py <源目录>` → push。

私有仓库，仅团队/买家可见。
